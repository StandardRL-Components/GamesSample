import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from collections import namedtuple
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# --- Data Structures ---
Plant = namedtuple('Plant', ['owner', 'ptype', 'health', 'max_health', 'angle', 'cooldown', 'growth'])
Particle = namedtuple('Particle', ['pos', 'vel', 'radius', 'color', 'lifespan'])
VineAttack = namedtuple('VineAttack', ['owner', 'start_pos', 'end_pos', 'progress', 'ptype'])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Conquer a star system by growing and evolving powerful plants on planets. "
        "Terraform worlds and launch vine attacks to defeat your rival."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select planets. Press space to attack or terraform to water. "
        "Hold shift to evolve plants or terraform to earth."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)

        # --- Game Constants ---
        self.MAX_STEPS = 5000
        self.NUM_PLANETS = 5
        self.TERRAFORM_COST = 40
        self.RESOURCE_GAIN_INTERVAL = 25
        self.ENEMY_DIFFICULTY_INTERVAL = 500

        # --- Colors ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_DARK = (0, 100, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_DARK = (100, 20, 20)
        self.COLOR_NEUTRAL = (128, 128, 128)
        self.COLOR_NEUTRAL_DARK = (70, 70, 70)
        self.COLOR_WATER = (50, 150, 255)
        self.COLOR_EARTH = (139, 69, 19)
        self.COLOR_VINE = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WHITE = (255, 255, 255)

        # --- Plant Types ---
        self.PLANT_TYPES = {
            'spore': {'max_health': 50, 'damage': 1.0, 'cooldown': 60, 'growth_rate': 0.005},
            'viner': {'max_health': 70, 'damage': 1.2, 'cooldown': 50, 'growth_rate': 0.007},
            'thorn': {'max_health': 100, 'damage': 2.0, 'cooldown': 45, 'growth_rate': 0.004}
        }
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = None
        
        self.planets = []
        self.particles = []
        self.vine_attacks = []
        self.stars = []
        
        self.player_resources = 0
        self.enemy_resources = 0
        self.player_unlocked_tier = 'spore'
        
        self.selected_planet_idx = 0
        self.current_step_reward = 0.0
        self.enemy_growth_modifier = 1.0

        self._create_stars()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = None
        
        self.player_resources = 50
        self.enemy_resources = 20
        self.player_unlocked_tier = 'spore'
        self.enemy_growth_modifier = 1.0
        
        self.selected_planet_idx = 0
        
        self.particles.clear()
        self.vine_attacks.clear()
        
        self._create_planets()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.current_step_reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_actions(movement, space_held, shift_held)
        self._update_game_state()
        
        reward = self.current_step_reward
        terminated = self._check_termination()
        
        if terminated:
            if self.winner == 'player':
                reward += 100
                self.score += 100
            elif self.winner == 'enemy':
                reward -= 100
                self.score -= 100
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, movement, space_held, shift_held):
        # --- 1. Planet Selection ---
        if movement != 0:
            # Debounce selection
            if not hasattr(self, '_last_move_step') or self.steps > self._last_move_step + 5:
                if movement in [1, 4]: # Up, Right -> Next
                    self.selected_planet_idx = (self.selected_planet_idx + 1) % self.NUM_PLANETS
                elif movement in [2, 3]: # Down, Left -> Previous
                    self.selected_planet_idx = (self.selected_planet_idx - 1 + self.NUM_PLANETS) % self.NUM_PLANETS
                self._last_move_step = self.steps

        selected_planet = self.planets[self.selected_planet_idx]
        if selected_planet['owner'] != 'player':
            return

        # --- 2. Primary Action (Space) ---
        if space_held:
            # Terraform to Water
            if self.player_resources >= self.TERRAFORM_COST and selected_planet['terrain'] != 'water':
                self.player_resources -= self.TERRAFORM_COST
                selected_planet['terrain'] = 'water'
                self.current_step_reward += 0.5
                self._create_particles(selected_planet['pos'], 30, self.COLOR_WATER)
            # Attack
            else:
                self._launch_vine_attack(selected_planet)

        # --- 3. Secondary Action (Shift) ---
        if shift_held:
            # Terraform to Earth
            if self.player_resources >= self.TERRAFORM_COST and selected_planet['terrain'] != 'earth':
                self.player_resources -= self.TERRAFORM_COST
                selected_planet['terrain'] = 'earth'
                self.current_step_reward += 0.5
                self._create_particles(selected_planet['pos'], 30, self.COLOR_EARTH)
            # Evolve
            else:
                self._evolve_plants(selected_planet)

    def _update_game_state(self):
        # Resource gain
        if self.steps % self.RESOURCE_GAIN_INTERVAL == 0:
            player_planets = sum(1 for p in self.planets if p['owner'] == 'player')
            enemy_planets = sum(1 for p in self.planets if p['owner'] == 'enemy')
            self.player_resources += player_planets
            self.enemy_resources += enemy_planets

        # Update difficulty
        if self.steps > 0 and self.steps % self.ENEMY_DIFFICULTY_INTERVAL == 0:
            self.enemy_growth_modifier += 0.05

        # Update unlocks
        player_planet_count = sum(1 for p in self.planets if p['owner'] == 'player')
        if player_planet_count >= 4 and self.player_unlocked_tier != 'thorn':
            self.player_unlocked_tier = 'thorn'
        elif player_planet_count >= 2 and self.player_unlocked_tier == 'spore':
            self.player_unlocked_tier = 'viner'

        # Update Planets and Plants
        for p in self.planets:
            self._update_planet(p)
            self._update_ownership(p)

        # Update Vines
        self._update_vines()
        
        # Update Particles
        updated_particles = []
        for p in self.particles:
            new_lifespan = p.lifespan - 1
            if new_lifespan > 0:
                new_pos = (p.pos[0] + p.vel[0], p.pos[1] + p.vel[1])
                updated_particles.append(p._replace(pos=new_pos, lifespan=new_lifespan))
        self.particles = updated_particles

    def _update_planet(self, planet):
        # Grow plants
        new_plants = []
        for plant in planet['plants']:
            if plant.growth < 1.0:
                growth_rate = self.PLANT_TYPES[plant.ptype]['growth_rate']
                if plant.owner == 'enemy':
                    growth_rate *= self.enemy_growth_modifier
                new_growth = min(1.0, plant.growth + growth_rate)
                plant = plant._replace(growth=new_growth)
            new_plants.append(plant)
        planet['plants'] = new_plants
        
        # Spawn new enemy plants if resources allow
        if planet['owner'] == 'enemy' and self.enemy_resources >= 20 and len(planet['plants']) < 8:
            if self.np_random.random() < 0.01:
                self.enemy_resources -= 20
                self._add_plant(planet, 'enemy', 'spore')

        # Intra-planet combat
        player_plants = [pl for pl in planet['plants'] if pl.owner == 'player']
        enemy_plants = [pl for pl in planet['plants'] if pl.owner == 'enemy']

        if not player_plants or not enemy_plants:
            return

        all_plants = planet['plants']
        for i, plant in enumerate(all_plants):
            new_cooldown = max(0, plant.cooldown - 1)
            all_plants[i] = plant._replace(cooldown=new_cooldown)

            if new_cooldown == 0:
                target_list = enemy_plants if plant.owner == 'player' else player_plants
                if target_list:
                    target_idx_in_list = self.np_random.integers(0, len(target_list))
                    target_plant = target_list[target_idx_in_list]
                    
                    for j, p_orig in enumerate(all_plants):
                        if p_orig is target_plant:
                            damage = self.PLANT_TYPES[plant.ptype]['damage']
                            new_health = p_orig.health - damage
                            all_plants[j] = p_orig._replace(health=new_health)
                            
                            if plant.owner == 'player':
                                self.current_step_reward += 0.1
                            
                            pos = self._get_plant_pos(planet, p_orig)
                            self._create_particles(pos, 3, self.COLOR_VINE)
                            break
                    
                    all_plants[i] = all_plants[i]._replace(cooldown=self.PLANT_TYPES[plant.ptype]['cooldown'])
        
        planet['plants'] = [p for p in all_plants if p.health > 0]

    def _update_ownership(self, planet):
        player_plants = any(p.owner == 'player' for p in planet['plants'])
        enemy_plants = any(p.owner == 'enemy' for p in planet['plants'])
        prev_owner = planet['owner']

        if player_plants and not enemy_plants:
            planet['owner'] = 'player'
            if prev_owner != 'player':
                self.current_step_reward += 5.0
        elif enemy_plants and not player_plants:
            planet['owner'] = 'enemy'
        elif not player_plants and not enemy_plants:
            planet['owner'] = 'neutral'
    
    def _update_vines(self):
        next_vines = []
        for vine in self.vine_attacks:
            new_progress = vine.progress + 0.02
            if new_progress >= 1.0:
                target_planet = None
                for p in self.planets:
                    if p['pos'] == vine.end_pos:
                        target_planet = p
                        break
                if target_planet:
                    self._create_particles(vine.end_pos, 20, self.COLOR_VINE)
                    if target_planet['owner'] != 'player':
                        if target_planet['plants'] and any(p.owner == 'enemy' for p in target_planet['plants']):
                            # Damage existing enemy plants
                            plants_to_damage = [p for p in target_planet['plants'] if p.owner == 'enemy']
                            idx = self.np_random.integers(0, len(plants_to_damage))
                            p_to_dmg = plants_to_damage[idx]
                            orig_idx = target_planet['plants'].index(p_to_dmg)
                            dmg = self.PLANT_TYPES[vine.ptype]['damage'] * 5
                            target_planet['plants'][orig_idx] = p_to_dmg._replace(health=p_to_dmg.health - dmg)
                            self.current_step_reward += 0.2
                        else: # Neutral or Player-owned but empty
                           if len(target_planet['plants']) < 8:
                               self._add_plant(target_planet, 'player', vine.ptype, initial_growth=0.5)
            else:
                next_vines.append(vine._replace(progress=new_progress))
        self.vine_attacks = next_vines

    def _check_termination(self):
        if self.game_over:
            return True
            
        player_owns_all = all(p['owner'] == 'player' for p in self.planets)
        player_wiped_out = all(p['owner'] != 'player' for p in self.planets) and \
                           all(len(p['plants']) == 0 for p in self.planets if p['owner'] == 'player') and \
                           not any(v.owner == 'player' for v in self.vine_attacks)

        if player_owns_all:
            self.game_over = True
            self.winner = 'player'
            return True
        if player_wiped_out:
            self.game_over = True
            self.winner = 'enemy'
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.winner = 'draw'
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_resources": self.player_resources,
            "player_planets": sum(1 for p in self.planets if p['owner'] == 'player'),
            "enemy_planets": sum(1 for p in self.planets if p['owner'] == 'enemy'),
            "player_unlocked_tier": self.player_unlocked_tier
        }
        
    def _render_game(self):
        # Render stars
        for x, y, size, alpha in self.stars:
            color = (alpha, alpha, alpha)
            if size > 0:
                pygame.draw.circle(self.screen, color, (x, y), size)

        # Render vine attacks
        for vine in self.vine_attacks:
            current_pos = (
                vine.start_pos[0] + (vine.end_pos[0] - vine.start_pos[0]) * vine.progress,
                vine.start_pos[1] + (vine.end_pos[1] - vine.start_pos[1]) * vine.progress
            )
            pygame.draw.line(self.screen, self.COLOR_VINE, vine.start_pos, current_pos, 3)
            pygame.draw.circle(self.screen, self.COLOR_VINE, (int(current_pos[0]), int(current_pos[1])), 5)

        # Render planets
        for i, p in enumerate(self.planets):
            owner_color_map = {'player': self.COLOR_PLAYER, 'enemy': self.COLOR_ENEMY, 'neutral': self.COLOR_NEUTRAL}
            terrain_color_map = {'normal': owner_color_map[p['owner']], 'water': self.COLOR_WATER, 'earth': self.COLOR_EARTH}
            color = terrain_color_map[p['terrain']]
            pygame.draw.circle(self.screen, color, (p['pos'][0], p['pos'][1]), p['radius'])
            
            if i == self.selected_planet_idx:
                pulse = int(1 + 3 * (1 + math.sin(self.steps * 0.2)) / 2)
                pygame.draw.circle(self.screen, self.COLOR_WHITE, p['pos'], p['radius'] + 5 + pulse, 2)

            for plant in p['plants']:
                self._render_plant(p, plant)
        
        for part in self.particles:
            pygame.draw.circle(self.screen, part.color, (int(part.pos[0]), int(part.pos[1])), int(part.radius))

    def _render_plant(self, planet, plant):
        pos = self._get_plant_pos(planet, plant)
        size = int(3 + 7 * plant.growth)
        color = self.COLOR_PLAYER if plant.owner == 'player' else self.COLOR_ENEMY

        pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), size)
        
        bar_width = 20
        bar_height = 4
        bar_pos_x = pos[0] - bar_width / 2
        bar_pos_y = pos[1] - size - bar_height - 2
        health_ratio = plant.health / plant.max_health
        
        pygame.draw.rect(self.screen, self.COLOR_ENEMY_DARK, (bar_pos_x, bar_pos_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_pos_x, bar_pos_y, bar_width * health_ratio, bar_height))

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        steps_surf = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_surf, (self.screen_width - steps_surf.get_width() - 10, 10))
        
        res_surf = self.font_ui.render(f"RESOURCES: {self.player_resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_surf, (10, 35))

        p = self.planets[self.selected_planet_idx]
        p_info = f"Selected: Planet {self.selected_planet_idx + 1} ({p['owner'].upper()})"
        p_surf = self.font_ui.render(p_info, True, self.COLOR_UI_TEXT)
        self.screen.blit(p_surf, (self.screen_width // 2 - p_surf.get_width() // 2, 10))

        action_text = "ACTION: [SPACE] Attack/Terra-Water | [SHIFT] Evolve/Terra-Earth"
        action_surf = self.font_ui.render(action_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(action_surf, (self.screen_width // 2 - action_surf.get_width() // 2, self.screen_height - 30))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "SYSTEM CONQUERED" if self.winner == 'player' else "ALL IS LOST"
            if self.winner == 'draw': win_text = "STALEMATE"
            
            title_surf = self.font_title.render(win_text, True, self.COLOR_WHITE)
            self.screen.blit(title_surf, (self.screen_width // 2 - title_surf.get_width() // 2, self.screen_height // 2 - 40))
            
            final_score_surf = self.font_ui.render(f"Final Score: {int(self.score)}", True, self.COLOR_WHITE)
            self.screen.blit(final_score_surf, (self.screen_width // 2 - final_score_surf.get_width() // 2, self.screen_height // 2 + 10))

    def _create_planets(self):
        self.planets.clear()
        positions = [
            (120, 100), (520, 100), (320, 200), (120, 300), (520, 300)
        ]
        radii = [40, 45, 60, 45, 40]
        
        owners = ['player', 'enemy'] + ['neutral'] * (self.NUM_PLANETS - 2)
        self.np_random.shuffle(owners)

        for i in range(self.NUM_PLANETS):
            planet = {
                'pos': positions[i],
                'radius': radii[i],
                'owner': owners[i],
                'terrain': 'normal',
                'plants': []
            }
            if owners[i] == 'player':
                self.selected_planet_idx = i
                self._add_plant(planet, 'player', 'spore')
            elif owners[i] == 'enemy':
                self._add_plant(planet, 'enemy', 'spore')
            self.planets.append(planet)

    def _add_plant(self, planet, owner, ptype, initial_growth=0.1):
        if len(planet['plants']) >= 8: return
        
        plant_info = self.PLANT_TYPES[ptype]
        new_plant = Plant(
            owner=owner,
            ptype=ptype,
            health=plant_info['max_health'],
            max_health=plant_info['max_health'],
            angle=self.np_random.uniform(0, 2 * math.pi),
            cooldown=plant_info['cooldown'],
            growth=initial_growth
        )
        planet['plants'].append(new_plant)
    
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(1, 4)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _get_plant_pos(self, planet, plant):
        radius_offset = planet['radius'] + 2
        return (
            planet['pos'][0] + math.cos(plant.angle) * radius_offset,
            planet['pos'][1] + math.sin(plant.angle) * radius_offset
        )

    def _launch_vine_attack(self, source_planet):
        if not any(p.ptype in ['viner', 'thorn'] for p in source_planet['plants']):
            return

        targets = [p for p in self.planets if p['owner'] != 'player' and p is not source_planet]
        if not targets: return

        target_planet = self.np_random.choice(targets)
        
        launching_plant_type = 'viner'
        if any(p.ptype == 'thorn' for p in source_planet['plants']):
            launching_plant_type = 'thorn'

        self.vine_attacks.append(VineAttack(
            owner='player',
            start_pos=source_planet['pos'],
            end_pos=target_planet['pos'],
            progress=0.0,
            ptype=launching_plant_type
        ))

    def _evolve_plants(self, planet):
        if self.player_unlocked_tier == 'spore': return

        made_change = False
        for i, plant in enumerate(planet['plants']):
            if plant.owner == 'player' and self.PLANT_TYPES[plant.ptype]['growth_rate'] < self.PLANT_TYPES[self.player_unlocked_tier]['growth_rate']:
                new_info = self.PLANT_TYPES[self.player_unlocked_tier]
                planet['plants'][i] = plant._replace(
                    ptype=self.player_unlocked_tier,
                    max_health=new_info['max_health'],
                    health=new_info['max_health']
                )
                made_change = True
        
        if made_change:
            self._create_particles(planet['pos'], 40, self.COLOR_PLAYER)

    def _create_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.screen_width)
            y = self.np_random.integers(0, self.screen_height)
            size = self.np_random.choice([0, 1, 1, 1, 2])
            alpha = self.np_random.integers(50, 150)
            self.stars.append((x, y, size, alpha))

if __name__ == '__main__':
    # This block is for manual play and will not be run by the evaluation system.
    # It requires pygame to be installed with display support.
    try:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        pygame.display.set_caption("Planet Vinea - Manual Control")
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        clock = pygame.time.Clock()
        running = True
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Info: {info}")
                pygame.time.wait(3000)
                obs, info = env.reset()
                
            clock.tick(30)

        pygame.quit()
    except pygame.error as e:
        print(f"Could not run main block, likely because of headless mode: {e}")