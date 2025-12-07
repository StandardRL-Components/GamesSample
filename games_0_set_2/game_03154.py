
# Generated: 2025-08-27T22:31:33.696904
# Source Brief: brief_03154.md
# Brief Index: 3154

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move the placement cursor. Shift to cycle tower types. Space to place a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Defend your base from waves of aliens by placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 7
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = 60

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_BASE = (0, 150, 100)
        self.COLOR_LANDING_ZONE = (100, 20, 30)
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_PROJECTILE = (50, 180, 255)
        self.COLOR_EXPLOSION = (255, 150, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_BG = (25, 30, 45)

        # Tower definitions
        self.TOWER_TYPES = [
            {'name': 'Basic', 'cost': 10, 'range': 80, 'damage': 1, 'fire_rate': 30, 'color': (255, 200, 0)},
            {'name': 'Fast', 'cost': 15, 'range': 70, 'damage': 0.6, 'fire_rate': 15, 'color': (0, 255, 255)},
            {'name': 'Range', 'cost': 20, 'range': 120, 'damage': 1.2, 'fire_rate': 40, 'color': (200, 0, 255)},
            {'name': 'Damage', 'cost': 25, 'range': 80, 'damage': 3, 'fire_rate': 60, 'color': (255, 100, 0)},
        ]
        
        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 25

        self.wave = 1
        self.max_waves = 5
        self.wave_transition_timer = 120 # 4 seconds at 30fps
        self.wave_cleared_bonus_given = False

        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0

        self.space_was_held = False
        self.shift_was_held = False
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if not self.game_over and not self.game_won:
            self._handle_input(movement, space_held, shift_held)
            
            # Place tower and get potential reward/cost
            reward += self._place_tower(space_held)

            # Update game state if not in wave transition
            if self.wave_transition_timer <= 0:
                self._update_aliens()
                reward += self._update_projectiles()
                self._update_towers()
            else:
                self.wave_transition_timer -= 1
                if self.wave_transition_timer == 0:
                    self.wave += 1
                    if self.wave > self.max_waves:
                        self.game_won = True
                    else:
                        self._spawn_wave()
                        self.wave_cleared_bonus_given = False

            self._update_particles()

            # Check for wave clear
            if len(self.aliens) == 0 and not self.game_won and not self.wave_cleared_bonus_given:
                reward += 50
                self.wave_cleared_bonus_given = True
                if self.wave < self.max_waves:
                    self.wave_transition_timer = 90 # 3 seconds
                else:
                    self.game_won = True
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.game_won:
            reward += 100

        # Clamp rewards
        reward = np.clip(reward, -100, 100)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1) # Right

        # Cycle tower type on key press (not hold)
        if shift_held and not self.shift_was_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.shift_was_held = shift_held
    
    def _place_tower(self, space_held):
        if space_held and not self.space_was_held:
            col, row = self.cursor_pos
            tower_def = self.TOWER_TYPES[self.selected_tower_type]
            
            if self.grid[row, col] == 0 and self.resources >= tower_def['cost']:
                self.resources -= tower_def['cost']
                world_x = self.GRID_OFFSET_X + col * self.CELL_SIZE + self.CELL_SIZE // 2
                world_y = self.GRID_OFFSET_Y + row * self.CELL_SIZE + self.CELL_SIZE // 2
                
                self.towers.append({
                    'x': world_x, 'y': world_y,
                    'type': self.selected_tower_type,
                    'cooldown': 0,
                    'flash': 0,
                })
                self.grid[row, col] = 1 # Mark as occupied
                # sound: place_tower.wav
                return 0 # No direct reward for placement, cost is the penalty
        self.space_was_held = space_held
        return 0

    def _update_aliens(self):
        to_remove = []
        for i, alien in enumerate(self.aliens):
            speed = alien['speed'] * (1 + (self.wave - 1) * 0.05)
            alien['y'] += speed
            if alien['y'] > self.HEIGHT - 40: # Reached landing zone
                self.base_health = max(0, self.base_health - alien['health'])
                to_remove.append(i)
                # sound: base_damage.wav
                # No reward here, handled in step loop to avoid double counting

        # Remove aliens that reached the base
        for i in sorted(to_remove, reverse=True):
            del self.aliens[i]

    def _update_projectiles(self):
        reward = 0
        proj_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['x'] += p['vx']
            p['y'] += p['vy']

            if not (0 < p['x'] < self.WIDTH and 0 < p['y'] < self.HEIGHT):
                proj_to_remove.append(i)
                continue

            alien_to_remove = -1
            for j, a in enumerate(self.aliens):
                dist = math.hypot(p['x'] - a['x'], p['y'] - a['y'])
                if dist < 10: # Hitbox
                    a['health'] -= p['damage']
                    reward += 0.1 # Reward for hit
                    self._create_explosion(p['x'], p['y'], 10, 5)
                    proj_to_remove.append(i)
                    
                    if a['health'] <= 0:
                        reward += 1.0 # Reward for kill
                        self.score += 10
                        self.resources += 5
                        alien_to_remove = j
                        self._create_explosion(a['x'], a['y'], 20, 10)
                        # sound: alien_die.wav
                    else:
                        # sound: alien_hit.wav
                        pass
                    break # Projectile hits only one alien
            
            if alien_to_remove != -1:
                del self.aliens[alien_to_remove]
        
        for i in sorted(list(set(proj_to_remove)), reverse=True):
            del self.projectiles[i]
        
        return reward

    def _update_towers(self):
        for tower in self.towers:
            if tower['flash'] > 0:
                tower['flash'] -= 1
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            tower_def = self.TOWER_TYPES[tower['type']]
            target = None
            min_dist = tower_def['range']

            for alien in self.aliens:
                dist = math.hypot(tower['x'] - alien['x'], tower['y'] - alien['y'])
                if dist < min_dist:
                    min_dist = dist
                    target = alien
            
            if target:
                angle = math.atan2(target['y'] - tower['y'], target['x'] - tower['x'])
                speed = 8
                self.projectiles.append({
                    'x': tower['x'], 'y': tower['y'],
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'damage': tower_def['damage']
                })
                tower['cooldown'] = tower_def['fire_rate']
                tower['flash'] = 5
                # sound: tower_shoot.wav

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['growth']

    def _spawn_wave(self):
        num_aliens = 5 + (self.wave - 1) * 2
        base_health = 1 + (self.wave - 1)
        
        for _ in range(num_aliens):
            self.aliens.append({
                'x': random.randint(20, self.WIDTH - 20),
                'y': random.randint(-100, -20),
                'health': base_health,
                'max_health': base_health,
                'speed': 0.5
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.steps >= 1000:
            self.game_over = True
        return self.game_over or self.game_won

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Landing zone
        pygame.draw.rect(self.screen, self.COLOR_LANDING_ZONE, (0, self.HEIGHT - 40, self.WIDTH, 40))
        # Grid
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.WIDTH - self.GRID_OFFSET_X, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.HEIGHT - 40))

    def _render_entities(self):
        # Towers and ranges
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower['type']]
            pos = (int(tower['x']), int(tower['y']))
            # Range indicator
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(tower_def['range']), (50, 60, 80))
            # Tower body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, tower_def['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, tower_def['color'])
            # Firing flash
            if tower['flash'] > 0:
                flash_alpha = 100 + (tower['flash'] * 30)
                flash_color = (*tower_def['color'], flash_alpha)
                s = pygame.Surface((30, 30), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, 15, 15, 15, flash_color)
                self.screen.blit(s, (pos[0] - 15, pos[1] - 15))

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['x']), int(alien['y']))
            pygame.gfxdraw.filled_trigon(self.screen, pos[0], pos[1] - 8, pos[0] - 8, pos[1] + 8, pos[0] + 8, pos[1] + 8, self.COLOR_ALIEN)
            pygame.gfxdraw.aatrigon(self.screen, pos[0], pos[1] - 8, pos[0] - 8, pos[1] + 8, pos[0] + 8, pos[1] + 8, self.COLOR_ALIEN)

        # Projectiles
        for p in self.projectiles:
            pos = (int(p['x']), int(p['y']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                color = (*self.COLOR_EXPLOSION, alpha)
                s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, p['radius'], p['radius'], p['radius'], color)
                self.screen.blit(s, (int(p['x']) - p['radius'], int(p['y']) - p['radius']))

    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, 40))
        # Base Health
        health_text = self.font_medium.render(f"Base: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 8))
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 8))
        # Wave
        wave_text = self.font_medium.render(f"Wave: {self.wave}/{self.max_waves}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 8))

        # Bottom bar (Tower selection)
        bottom_bar_y = self.HEIGHT - 40
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, bottom_bar_y, self.WIDTH, 40))
        # Resources
        res_text = self.font_medium.render(f"${self.resources}", True, (255, 223, 0))
        self.screen.blit(res_text, (10, bottom_bar_y + 8))
        # Tower info
        start_x = 100
        for i, t_def in enumerate(self.TOWER_TYPES):
            box_x = start_x + i * 130
            is_selected = i == self.selected_tower_type
            
            # Highlight box
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (box_x - 5, bottom_bar_y + 5, 120, 30), 2, 3)

            # Tower icon
            pygame.gfxdraw.filled_circle(self.screen, box_x + 10, bottom_bar_y + 20, 8, t_def['color'])
            # Tower text
            name_text = self.font_small.render(f"{t_def['name']} (${t_def['cost']})", True, self.COLOR_TEXT)
            self.screen.blit(name_text, (box_x + 25, bottom_bar_y + 12))

        # Cursor
        cursor_x = self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = 100 + pulse * 155
        
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        color = (*self.COLOR_CURSOR, alpha)
        pygame.draw.rect(s, color, (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3)
        self.screen.blit(s, (cursor_x, cursor_y))

        # Game Over / Win / Wave Clear Messages
        if self.game_over:
            self._render_centered_text("GAME OVER", self.font_large, self.COLOR_ALIEN)
        elif self.game_won:
            self._render_centered_text("YOU WIN!", self.font_large, self.COLOR_BASE)
        elif self.wave_transition_timer > 0:
            self._render_centered_text(f"WAVE {self.wave} CLEARED", self.font_large, self.COLOR_TEXT)

    def _render_centered_text(self, text, font, color):
        text_surf = font.render(text, True, color)
        x = self.WIDTH // 2 - text_surf.get_width() // 2
        y = self.HEIGHT // 2 - text_surf.get_height() // 2
        
        # Simple shadow
        shadow_surf = font.render(text, True, (0,0,0))
        self.screen.blit(shadow_surf, (x + 2, y + 2))
        
        self.screen.blit(text_surf, (x, y))

    def _create_explosion(self, x, y, radius, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            life = random.randint(15, 30)
            self.particles.append({
                'x': x, 'y': y,
                'radius': random.uniform(1, 3),
                'growth': 0.2,
                'life': life, 'max_life': life,
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "aliens_remaining": len(self.aliens),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert info['base_health'] == 100
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Human Play ---
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard state for human control
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] or keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print("Game Over! Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
        
        # Control the frame rate
        clock.tick(30) # Run at 30 FPS

    env.close()