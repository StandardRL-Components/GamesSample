import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:18:04.742361
# Source Brief: brief_02695.md
# Brief Index: 2695
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your bastion by strategically placing clones to fight off incoming enemy waves. "
        "Use your cursor to deploy units and unleash a powerful magnetic pulse."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place the selected clone. Press shift to activate a magnetic pulse."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.BASTION_Y_LIMIT = 320

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PULSE = (255, 255, 255)
        self.ENERGY_BAR_COLOR = (60, 180, 255)
        
        # Game Mechanics
        self.CURSOR_SPEED = 8
        self.INITIAL_ENERGY = 100
        self.PASSIVE_ENERGY_REGEN = 0.1
        self.ENERGY_PER_KILL = 15
        self.PULSE_COST = 25
        self.PULSE_COOLDOWN = 30  # steps
        self.PULSE_RADIUS = 120
        self.PULSE_DAMAGE = 20
        self.WAVE_PREP_TIME = 150 # steps between waves

        # Clone Specifications
        self.CLONE_SPECS = [
            {'name': 'Sentinel', 'cost': 50, 'hp': 100, 'radius': 12, 'color': (50, 200, 255)},
            {'name': 'Bulwark', 'cost': 80, 'hp': 250, 'radius': 16, 'color': (200, 100, 255)},
            {'name': 'Nexus', 'cost': 120, 'hp': 80, 'radius': 10, 'color': (50, 255, 150)},
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.energy = 0
        self.wave_number = 0
        self.clones = []
        self.enemies = []
        self.particles = []
        self.pulse_effects = []
        self.unlocked_clone_types = []
        self.selected_clone_type_idx = 0
        self.wave_in_progress = False
        self.next_wave_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.pulse_cooldown_timer = 0
        self.min_clone_cost = min(spec['cost'] for spec in self.CLONE_SPECS)

        # Initialize state by calling reset
        self.reset()
        
        # --- Validation ---
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.energy = self.INITIAL_ENERGY
        self.wave_number = 0
        
        self.clones = []
        self.enemies = []
        self.particles = []
        self.pulse_effects = []
        
        self.unlocked_clone_types = {0}
        self.selected_clone_type_idx = 0
        
        self.wave_in_progress = False
        self.next_wave_timer = self.WAVE_PREP_TIME // 2

        self.prev_space_held = False
        self.prev_shift_held = False
        self.pulse_cooldown_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Update Timers and Resources ---
        self.pulse_cooldown_timer = max(0, self.pulse_cooldown_timer - 1)
        self.energy = min(999, self.energy + self.PASSIVE_ENERGY_REGEN)

        # --- Handle Input ---
        reward += self._handle_input(action)
        
        # --- Update Game Logic ---
        reward += self._update_clones()
        reward += self._update_enemies()
        self._update_effects()
        
        # --- Wave Management ---
        if not self.wave_in_progress and not self.enemies:
            self.next_wave_timer -= 1
            if self.next_wave_timer <= 0:
                reward += self._start_new_wave()

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.steps >= self.MAX_STEPS:
                reward += 50.0  # Win bonus
            else:
                reward += -100.0 # Loss penalty
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        reward = 0.0
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1
        
        # 1. Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # 2. Place Clone (Space)
        space_press = space_held and not self.prev_space_held
        if space_press:
            clone_spec = self.CLONE_SPECS[self.selected_clone_type_idx]
            if self.energy >= clone_spec['cost'] and self.cursor_pos[1] < self.BASTION_Y_LIMIT:
                self.energy -= clone_spec['cost']
                self.clones.append({
                    'pos': self.cursor_pos.copy(),
                    'type_idx': self.selected_clone_type_idx,
                    'hp': clone_spec['hp'],
                    'max_hp': clone_spec['hp'],
                    'radius': clone_spec['radius'],
                    'anim_timer': random.uniform(0, 2 * math.pi)
                })
                # Sound: clone_place.wav

        # 3. Magnetic Pulse (Shift)
        shift_press = shift_held and not self.prev_shift_held
        if shift_press and self.pulse_cooldown_timer == 0 and self.energy >= self.PULSE_COST:
            self.energy -= self.PULSE_COST
            self.pulse_cooldown_timer = self.PULSE_COOLDOWN
            self.pulse_effects.append({
                'pos': self.cursor_pos.copy(),
                'radius': 0,
                'max_radius': self.PULSE_RADIUS,
                'lifespan': 20,
                'hit_enemies': set()
            })
            # Sound: magnetic_pulse.wav
            
            for enemy in self.enemies:
                dist = np.linalg.norm(self.cursor_pos - enemy['pos'])
                if dist < self.PULSE_RADIUS:
                    enemy['hp'] -= self.PULSE_DAMAGE
                    reward += 0.1 # Reward for damaging
                    self._create_particles(enemy['pos'], 3, self.COLOR_PULSE)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        self.next_wave_timer = self.WAVE_PREP_TIME

        # Unlock new clones
        if self.wave_number % 5 == 0:
            new_type = self.wave_number // 5
            if new_type < len(self.CLONE_SPECS):
                self.unlocked_clone_types.add(new_type)
        
        # Auto-select best available clone
        self.selected_clone_type_idx = max(self.unlocked_clone_types)

        # Spawn enemies
        num_enemies = 2 + self.wave_number
        enemy_speed = 0.5 + self.wave_number * 0.05
        enemy_health = 10 + self.wave_number * 1.0
        
        for _ in range(num_enemies):
            self.enemies.append({
                'pos': np.array([random.uniform(20, self.SCREEN_WIDTH - 20), random.uniform(-80, -20)], dtype=np.float32),
                'hp': enemy_health,
                'max_hp': enemy_health,
                'speed': enemy_speed,
                'radius': 8,
                'target_clone': None
            })
        
        return 10.0 # Reward for surviving a wave

    def _update_clones(self):
        reward = 0.0
        clones_to_remove = []
        for i, clone in enumerate(self.clones):
            clone['anim_timer'] += 0.1
            if clone['hp'] <= 0:
                reward -= 5.0 # Penalty for losing a clone
                clones_to_remove.append(i)
                spec = self.CLONE_SPECS[clone['type_idx']]
                self._create_particles(clone['pos'], 30, spec['color'])
                # Sound: clone_destroyed.wav
        
        if clones_to_remove:
            self.clones = [c for i, c in enumerate(self.clones) if i not in clones_to_remove]
        
        return reward

    def _update_enemies(self):
        reward = 0.0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy['hp'] <= 0:
                reward += 1.0 # Reward for destroying an enemy
                self.energy += self.ENERGY_PER_KILL
                enemies_to_remove.append(i)
                self._create_particles(enemy['pos'], 15, self.COLOR_ENEMY)
                # Sound: enemy_destroyed.wav
                continue

            # Find closest clone to target
            closest_clone = None
            min_dist = float('inf')
            if self.clones:
                for clone in self.clones:
                    dist = np.linalg.norm(enemy['pos'] - clone['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_clone = clone
            
            # Movement
            if closest_clone:
                direction = closest_clone['pos'] - enemy['pos']
                dist_to_clone = np.linalg.norm(direction)
                if dist_to_clone > 0:
                    direction /= dist_to_clone

                # Collision check
                if dist_to_clone < enemy['radius'] + closest_clone['radius']:
                    damage = 1
                    closest_clone['hp'] -= damage
                    enemy['hp'] -= damage * 2 # Clones are tough
                    reward -= 0.01 * damage # Penalty for clone taking damage
                    # Create small spark on impact
                    self._create_particles(enemy['pos'], 1, self.COLOR_TEXT, 0.5)
                else:
                    enemy['pos'] += direction * enemy['speed']
            else: # No clones, move towards bastion
                direction = np.array([0, 1])
                enemy['pos'] += direction * enemy['speed']

            # If enemy reaches bastion, it disappears (counts as a leak, but no penalty for now)
            if enemy['pos'][1] > self.BASTION_Y_LIMIT:
                enemies_to_remove.append(i)

        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

        if not self.enemies and self.wave_in_progress:
            self.wave_in_progress = False

        return reward

    def _update_effects(self):
        # Update particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['vel'][1] += 0.05 # gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(i)
        if particles_to_remove:
            self.particles = [p for i, p in enumerate(self.particles) if i not in particles_to_remove]

        # Update pulse effects
        pulses_to_remove = []
        for i, p in enumerate(self.pulse_effects):
            p['radius'] += p['max_radius'] / p['lifespan']
            p['lifespan'] -= 0.5
            if p['lifespan'] <= 0:
                pulses_to_remove.append(i)
        if pulses_to_remove:
            self.pulse_effects = [p for i, p in enumerate(self.pulse_effects) if i not in pulses_to_remove]
    
    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'lifespan': random.randint(20, 40),
                'color': color,
                'size': random.uniform(1, 3)
            })

    def _check_termination(self):
        # Loss condition: no clones and not enough energy to place a new one, after wave 1 started
        if self.wave_number > 0 and not self.clones and self.energy < self.min_clone_cost:
            return True
        # Win condition: max steps reached
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        pygame.draw.line(self.screen, self.ENERGY_BAR_COLOR, (0, self.BASTION_Y_LIMIT), (self.SCREEN_WIDTH, self.BASTION_Y_LIMIT), 2)

        # Draw clones
        for clone in self.clones:
            spec = self.CLONE_SPECS[clone['type_idx']]
            pos_int = clone['pos'].astype(int)
            # Pulsing glow
            glow_radius = spec['radius'] + 4 + 2 * math.sin(clone['anim_timer'])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(glow_radius), (*spec['color'], 50))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(glow_radius), (*spec['color'], 50))
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], spec['radius'], spec['color'])
            # Health bar
            if clone['hp'] < clone['max_hp']:
                hp_ratio = clone['hp'] / clone['max_hp']
                bar_width = spec['radius'] * 2
                bar_height = 4
                bar_x = pos_int[0] - spec['radius']
                bar_y = pos_int[1] + spec['radius'] + 4
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (50, 255, 50), (bar_x, bar_y, int(bar_width * hp_ratio), bar_height))

        # Draw enemies
        for enemy in self.enemies:
            pos_int = enemy['pos'].astype(int)
            r = int(enemy['radius'])
            points = [
                (pos_int[0], pos_int[1] - r),
                (pos_int[0] + r, pos_int[1] + r),
                (pos_int[0] - r, pos_int[1] + r),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Draw pulse effects
        for p in self.pulse_effects:
            alpha = max(0, 255 * (p['lifespan'] / 10))
            alpha = min(255, alpha) # Clamp alpha to valid 0-255 range
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*self.COLOR_PULSE, int(alpha)))

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 40))
            color = (*p['color'], int(alpha))
            pygame.draw.circle(self.screen, color, p['pos'].astype(int), int(p['size']))

    def _render_ui(self):
        # Wave counter
        wave_text = self.font_large.render(f"WAVE {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

        # Energy bar
        energy_label = self.font_large.render("ENERGY", True, self.COLOR_TEXT)
        self.screen.blit(energy_label, (self.SCREEN_WIDTH - 120, self.SCREEN_HEIGHT - 40))
        energy_val_text = self.font_large.render(f"{int(self.energy)}", True, self.ENERGY_BAR_COLOR)
        self.screen.blit(energy_val_text, (self.SCREEN_WIDTH - energy_val_text.get_width() - 10, self.SCREEN_HEIGHT - 70))

        # Selected Clone UI
        spec = self.CLONE_SPECS[self.selected_clone_type_idx]
        clone_label = self.font_small.render("SELECTED CLONE", True, self.COLOR_TEXT)
        self.screen.blit(clone_label, (10, self.SCREEN_HEIGHT - 70))
        clone_name = self.font_large.render(spec['name'], True, spec['color'])
        self.screen.blit(clone_name, (10, self.SCREEN_HEIGHT - 50))
        clone_cost = self.font_small.render(f"Cost: {spec['cost']} E", True, self.COLOR_TEXT)
        self.screen.blit(clone_cost, (10, self.SCREEN_HEIGHT - 25))

        # Cursor
        cursor_pos_int = self.cursor_pos.astype(int)
        c = self.COLOR_CURSOR
        l = 10 # length of cursor lines
        pygame.draw.line(self.screen, c, (cursor_pos_int[0] - l, cursor_pos_int[1]), (cursor_pos_int[0] + l, cursor_pos_int[1]))
        pygame.draw.line(self.screen, c, (cursor_pos_int[0], cursor_pos_int[1] - l), (cursor_pos_int[0], cursor_pos_int[1] + l))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "energy": self.energy,
            "clones": len(self.clones),
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Terraformed Bastion")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()