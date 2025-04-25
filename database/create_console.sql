

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


