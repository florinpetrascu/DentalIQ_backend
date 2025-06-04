

create table patients (

        id numeric,
        first_name varchar,
        last_name varchar,
        phone_number varchar

);

create table teeth (

    id numeric,
    name varchar,
    polygon varchar

);

create table notes(

    id numeric,
    text varchar,
    date timestamp

);

create table issues(

    id numeric,
    name varchar

);

alter table teeth add constraint un_tth_name unique (name);

-- set primary key

ALTER TABLE patients
    ALTER COLUMN id TYPE INTEGER;

ALTER TABLE teeth
    ALTER COLUMN id TYPE INTEGER;

ALTER TABLE issues
    ALTER COLUMN id TYPE INTEGER;

ALTER TABLE notes
    ALTER COLUMN id TYPE INTEGER;


ALTER TABLE patients
    ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY;

ALTER TABLE teeth
    ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY;

ALTER TABLE issues
    ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY;

ALTER TABLE notes
    ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY;

-- add foreign key

alter table teeth add column patient_id numeric;

alter table notes add column teeth_id numeric;

alter table issues add column teeth_id numeric;

-- schimbam la integer
ALTER TABLE teeth
    ALTER COLUMN patient_id TYPE INTEGER;

ALTER TABLE issues
    ALTER COLUMN teeth_id TYPE INTEGER;

ALTER TABLE notes
    ALTER COLUMN teeth_id TYPE INTEGER;


-- teeth → patients
ALTER TABLE teeth
ADD CONSTRAINT fk_teeth_patient
FOREIGN KEY (patient_id)
REFERENCES patients(id);

-- notes → teeth
ALTER TABLE notes
ADD CONSTRAINT fk_notes_teeth
FOREIGN KEY (teeth_id)
REFERENCES teeth(id);

-- issues → teeth
ALTER TABLE issues
ADD CONSTRAINT fk_issues_teeth
FOREIGN KEY (teeth_id)
REFERENCES teeth(id);


create table radiographs(

    id integer,
    url varchar,
    patient_id integer

);

ALTER TABLE notes
    ALTER COLUMN id TYPE INTEGER;

ALTER TABLE radiographs
ALTER COLUMN id SET NOT NULL;

ALTER TABLE radiographs
ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY;

-- radiographs → patients
ALTER TABLE radiographs
ADD CONSTRAINT fk_radiographs_patient
FOREIGN KEY (patient_id)
REFERENCES patients(id);


ALTER TABLE teeth DROP CONSTRAINT un_tth_name;

-- 3. Creezi unul nou cu ON DELETE CASCADE
ALTER TABLE issues
ADD CONSTRAINT fk_issues_teeth
FOREIGN KEY (teeth_id)
REFERENCES teeth(id)
ON DELETE CASCADE;

ALTER TABLE teeth
ADD CONSTRAINT fk_teeth_patient
FOREIGN KEY (patient_id)
REFERENCES patients(id)
ON DELETE CASCADE;

ALTER TABLE radiographs
ADD CONSTRAINT fk_radiographs_patient
FOREIGN KEY (patient_id)
REFERENCES patients(id)
ON DELETE CASCADE;

-- Creem tabela de medici

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    nume TEXT,
    prenume TEXT,
    username TEXT UNIQUE,
    hashed_password TEXT,
    email TEXT,
    activ BOOLEAN
);