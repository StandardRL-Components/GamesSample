import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the environment to headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move cursor. Space to place a standard tower, Shift to place a long-range tower."
    )

    game_description = (
        "Defend your base from waves of alien invaders by strategically placing defensive towers."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CELL_SIZE = 40
    GRID_W, GRID_H = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
    MAX_STEPS = 1000
    FPS = 30

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_BASE = (60, 180, 75)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (60, 180, 75)
    COLOR_HEALTH_BAR_BG = (70, 0, 0)

    # Entity Properties
    ALIEN_PROPS = {
        'red': {'health': 10, 'speed': 1.0, 'reward': 1, 'color': (230, 25, 75)},
        'blue': {'health': 5, 'speed': 2.0, 'reward': 1, 'color': (0, 130, 200)},
        'yellow': {'health': 30, 'speed': 0.7, 'reward': 3, 'color': (255, 225, 25)},
    }
    TOWER_PROPS = {
        'standard': {'range': 100, 'fire_rate': 30, 'damage': 5, 'color': (250, 190, 212)},
        'long_range': {'range': 200, 'fire_rate': 60, 'damage': 12, 'color': (170, 110, 40)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables to be populated in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_pos = None
        self.base_rect = None
        self.base_health = 0
        self.cursor_pos = None
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.occupied_cells = set()
        self.alien_spawn_timer = 0
        self.alien_spawn_rate = 90  # Lower is faster
        self.available_aliens = []
        self.reward_this_step = 0
        
        # self.reset() is called by the test harness, no need to call it here.
        # self.validate_implementation() is for debugging, not needed for standard init.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        base_grid_pos = (1, self.GRID_H // 2)
        self.base_pos = pygame.math.Vector2(
            base_grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            base_grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        )
        self.base_rect = pygame.Rect(
            base_grid_pos[0] * self.CELL_SIZE, base_grid_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        self.base_health = 100
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.aliens = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.occupied_cells = {base_grid_pos}
        
        self.alien_spawn_timer = 0
        self.alien_spawn_rate = 90
        self.available_aliens = ['red']
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if self.game_over:
            # After termination, we should still return a valid observation
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle player actions ---
        self._handle_input(action)
        
        # --- Update game logic ---
        self._update_spawner()
        self._update_towers()
        self._update_projectiles()
        self._update_aliens()
        self._handle_collisions()
        self._update_particles()
        
        # --- Update game state ---
        self.steps += 1
        self.reward_this_step -= 0.01  # Survival penalty

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated: # Win condition
            self.reward_this_step += 100
            self.game_over = True
        
        self.score += self.reward_this_step
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Place tower (Shift has priority)
        if shift_held:
            self._place_tower('long_range')
        elif space_held:
            self._place_tower('standard')

    def _place_tower(self, tower_type):
        pos_tuple = tuple(self.cursor_pos)
        if pos_tuple not in self.occupied_cells:
            # SFX: place_tower.wav
            props = self.TOWER_PROPS[tower_type]
            new_tower = {
                'pos': (pos_tuple[0] * self.CELL_SIZE + self.CELL_SIZE / 2, 
                        pos_tuple[1] * self.CELL_SIZE + self.CELL_SIZE / 2),
                'type': tower_type,
                'cooldown': 0,
                **props
            }
            self.towers.append(new_tower)
            self.occupied_cells.add(pos_tuple)

    def _update_spawner(self):
        # Increase difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.alien_spawn_rate = max(20, self.alien_spawn_rate * 0.9)
            if self.steps == 200 and 'blue' not in self.available_aliens:
                self.available_aliens.append('blue')
            if self.steps == 600 and 'yellow' not in self.available_aliens:
                self.available_aliens.append('yellow')
        
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            self.alien_spawn_timer = int(self.alien_spawn_rate)
            
            spawn_y = self.np_random.integers(0, self.HEIGHT)
            alien_type = self.np_random.choice(self.available_aliens)
            props = self.ALIEN_PROPS[alien_type]
            
            new_alien = {
                'pos': pygame.math.Vector2(self.WIDTH + 20, spawn_y),
                'type': alien_type,
                'target_pos': self.base_pos,
                'health': props['health'], # Explicitly copy health
                **props
            }
            self.aliens.append(new_alien)

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    # SFX: shoot.wav
                    self._create_projectile(tower, target)
                    tower['cooldown'] = tower['fire_rate']
                    # Muzzle flash particle
                    self._create_particle(tower['pos'], tower['color'], 5, 8, 5)

    def _find_target(self, tower):
        tower_pos = pygame.math.Vector2(tower['pos'])
        for alien in self.aliens:
            if tower_pos.distance_to(alien['pos']) <= tower['range']:
                return alien
        return None

    def _create_projectile(self, tower, target):
        tower_pos = pygame.math.Vector2(tower['pos'])
        direction = (target['pos'] - tower_pos).normalize()
        
        projectile = {
            'pos': tower_pos,
            'vel': direction * 8,
            'damage': tower['damage'],
            'lifespan': 60 # frames
        }
        self.projectiles.append(projectile)

    def _update_projectiles(self):
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.projectiles = [p for p in self.projectiles if p['lifespan'] > 0 and 0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT]

    def _update_aliens(self):
        for alien in self.aliens:
            if (alien['pos'] - alien['target_pos']).length() > alien['speed']:
                direction = (alien['target_pos'] - alien['pos']).normalize()
                alien['pos'] += direction * alien['speed']
            else:
                alien['pos'] = alien['target_pos'].copy()


    def _handle_collisions(self):
        # Projectiles vs Aliens
        for p in self.projectiles[:]:
            for alien in self.aliens[:]:
                if pygame.math.Vector2(p['pos']).distance_to(alien['pos']) < 15:
                    # SFX: hit_alien.wav
                    self.reward_this_step += 0.1
                    alien['health'] -= p['damage']
                    self._create_particle(alien['pos'], alien['color'], 10, 20, 10)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if alien['health'] <= 0:
                        # SFX: kill_alien.wav
                        self.reward_this_step += alien['reward']
                        self._create_particle(alien['pos'], (255, 255, 255), 15, 30, 15)
                        if alien in self.aliens: self.aliens.remove(alien)
                    break

        # Aliens vs Base
        for alien in self.aliens[:]:
            if self.base_rect.collidepoint(alien['pos']):
                # SFX: base_damage.wav
                self.base_health -= alien['health'] # Alien deals its remaining health as damage
                self.reward_this_step -= 5 # Penalty for base damage
                self._create_particle(alien['pos'], self.COLOR_BASE_DMG, 20, 40, 20)
                if alien in self.aliens: self.aliens.remove(alien)
        self.base_health = max(0, self.base_health)

    def _create_particle(self, pos, color, start_radius, end_radius, lifespan):
        self.particles.append({
            'pos': pygame.math.Vector2(pos),
            'start_radius': start_radius,
            'max_radius': end_radius,
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'color': color
        })

    def _update_particles(self):
        for p in self.particles:
            p['lifespan'] -= 1
            progress = (p['max_lifespan'] - p['lifespan']) / p['max_lifespan']
            p['radius'] = p['start_radius'] + (p['max_radius'] - p['start_radius']) * progress
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step -= 50 # Loss penalty
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_base()
        self._render_towers()
        self._render_aliens()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_base(self):
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        pygame.gfxdraw.rectangle(self.screen, self.base_rect, (*self.COLOR_BASE, 150))

    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, tower['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 3, tower['color'])

    def _render_aliens(self):
        for alien in self.aliens:
            pos = alien['pos']
            size = 10
            points = [
                (pos.x, pos.y - size),
                (pos.x - size / 1.5, pos.y + size / 2),
                (pos.x + size / 1.5, pos.y + size / 2),
            ]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, alien['color'])
            pygame.gfxdraw.filled_polygon(self.screen, int_points, alien['color'])

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.draw.rect(self.screen, (255, 255, 255), (pos[0]-2, pos[1]-2, 4, 4))

    def _render_particles(self):
        for p in self.particles:
            if 'radius' not in p: continue # Skip if not updated yet
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            radius_int = int(p['radius'])
            if radius_int <= 0: continue
            s = pygame.Surface((radius_int*2, radius_int*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius_int, radius_int), radius_int)
            self.screen.blit(s, (int(p['pos'].x - radius_int), int(p['pos'].y - radius_int)))

    def _render_cursor(self):
        if self.game_over: return
        
        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        is_valid = tuple(self.cursor_pos) not in self.occupied_cells
        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        
        pygame.draw.rect(self.screen, color, cursor_rect, 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / 100
        bar_width = 200
        bar_height = 20
        health_bar_rect = pygame.Rect(10, 10, int(bar_width * health_ratio), bar_height)
        health_bar_bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, health_bar_rect)
        
        # Step Count
        step_text = self.font_ui.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 10, 10))
        
        # Game Over Text
        if self.game_over:
            if self.base_health <= 0:
                msg = "BASE DESTROYED"
                color = self.COLOR_CURSOR_INVALID
            else:
                msg = "VICTORY"
                color = self.COLOR_BASE
            
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "aliens": len(self.aliens),
            "towers": len(self.towers),
        }

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy videodriver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display score on window title
        pygame.display.set_caption(f"Tower Defense | Score: {total_reward:.2f} | Health: {info['base_health']}")

        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()