
# Generated: 2025-08-28T02:07:29.767075
# Source Brief: brief_04340.md
# Brief Index: 4340

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ↑↓←→ to move the cursor. Space to place a Basic Tower. Shift to place an Advanced Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies by strategically placing defensive towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PATH = (60, 65, 70)
        self.COLOR_END_ZONE = (100, 20, 20)
        self.COLOR_CURSOR = (0, 200, 255, 150)
        self.COLOR_TOWER_BASIC = (0, 255, 128)
        self.COLOR_TOWER_ADVANCED = (100, 255, 200)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)

        # Game constants
        self.MAX_STEPS = 30 * 60 * 3 # 3 minutes at 30fps
        self.TOTAL_WAVES = 10
        self.INTER_WAVE_DURATION = 30 * 8 # 8 seconds
        
        self.TOWER_BASIC_COST = 10
        self.TOWER_BASIC_RANGE = 100
        self.TOWER_BASIC_DAMAGE = 35
        self.TOWER_BASIC_COOLDOWN = 30 # 1s
        
        self.TOWER_ADVANCED_COST = 20
        self.TOWER_ADVANCED_RANGE = 150
        self.TOWER_ADVANCED_DAMAGE = 50
        self.TOWER_ADVANCED_COOLDOWN = 45 # 1.5s
        
        # Define zombie path waypoints (in grid coordinates)
        self.path_waypoints_grid = [
            (-1, 2), (2, 2), (2, 7), (10, 7), (10, 3), (13, 3), (13, 8), (16, 8)
        ]
        self.path_waypoints_px = [(p[0] * self.GRID_SIZE + self.GRID_SIZE // 2, p[1] * self.GRID_SIZE + self.GRID_SIZE // 2) for p in self.path_waypoints_grid]

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.wave_number = 1
        self.resources = 30
        
        self.towers = []
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.cursor_move_cooldown = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.game_phase = "placement" # "placement" or "wave"
        self.phase_timer = self.INTER_WAVE_DURATION
        
        self.zombies_to_spawn = []
        self.zombie_spawn_timer = 0
        self.zombies_at_end = 0
        
        self._prepare_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Handle player actions
        placement_reward = self._handle_actions(action)
        reward += placement_reward
        
        # Update game logic
        self._update_phase_logic()
        
        killed_this_step = self._update_towers_and_projectiles()
        reward += killed_this_step * 0.1
        self.score += killed_this_step * 10

        ended_this_step = self._update_zombies()
        reward -= ended_this_step * 10
        self.zombies_at_end += ended_this_step

        self._update_particles()
        
        # Check for wave completion
        if self.game_phase == "wave" and not self.zombies and not self.zombies_to_spawn:
            if self.wave_number <= self.TOTAL_WAVES:
                reward += 1
                self.score += 100
                self.resources += 15 + self.wave_number * 2
                self.game_phase = "placement"
                self.phase_timer = self.INTER_WAVE_DURATION
                self.wave_number += 1
                if self.wave_number <= self.TOTAL_WAVES:
                    self._prepare_next_wave()

        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.zombies_at_end > 0:
            self.game_over = True
            terminated = True
            reward = -50
        elif self.wave_number > self.TOTAL_WAVES:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward = 50
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 10 # Timeout penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # Move cursor
        if self.cursor_move_cooldown > 0:
            self.cursor_move_cooldown -= 1
        elif movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)
            self.cursor_move_cooldown = 4 # 4 frames cooldown for movement

        # Place towers (only on key press, not hold)
        can_place = all(t['grid_pos'] != self.cursor_pos for t in self.towers)
        
        if space_held and not self.last_space_held and can_place:
            if self.resources >= self.TOWER_BASIC_COST:
                self.resources -= self.TOWER_BASIC_COST
                self.towers.append({
                    'grid_pos': list(self.cursor_pos),
                    'px_pos': (self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE//2, self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE//2),
                    'type': 'basic',
                    'cooldown': 0
                })
                reward += 0.2
                # SFX: place_tower.wav
                self._add_particles(self.towers[-1]['px_pos'], 10, self.COLOR_TOWER_BASIC, 5, 15)

        if shift_held and not self.last_shift_held and can_place:
            if self.resources >= self.TOWER_ADVANCED_COST:
                self.resources -= self.TOWER_ADVANCED_COST
                self.towers.append({
                    'grid_pos': list(self.cursor_pos),
                    'px_pos': (self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE//2, self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE//2),
                    'type': 'advanced',
                    'cooldown': 0
                })
                reward += 0.2
                # SFX: place_tower_advanced.wav
                self._add_particles(self.towers[-1]['px_pos'], 10, self.COLOR_TOWER_ADVANCED, 5, 15)

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _update_phase_logic(self):
        if self.game_phase == "placement":
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.game_phase = "wave"
        elif self.game_phase == "wave":
            if self.zombies_to_spawn:
                self.zombie_spawn_timer -= 1
                if self.zombie_spawn_timer <= 0:
                    self.zombies.append(self.zombies_to_spawn.pop(0))
                    self.zombie_spawn_timer = 30 # Spawn one zombie per second

    def _prepare_next_wave(self):
        num_zombies = 4 + self.wave_number
        speed = 1.0 + (self.wave_number - 1) * 0.05
        health = 100 + (self.wave_number - 1) * 10
        self.zombies_to_spawn = []
        for _ in range(num_zombies):
            self.zombies_to_spawn.append({
                'pos': list(self.path_waypoints_px[0]),
                'health': health,
                'max_health': health,
                'speed': speed,
                'waypoint_idx': 1
            })

    def _update_towers_and_projectiles(self):
        # Update towers
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            target = None
            min_dist = float('inf')
            
            is_basic = tower['type'] == 'basic'
            tower_range = self.TOWER_BASIC_RANGE if is_basic else self.TOWER_ADVANCED_RANGE
            
            for zombie in self.zombies:
                dist = math.hypot(tower['px_pos'][0] - zombie['pos'][0], tower['px_pos'][1] - zombie['pos'][1])
                if dist < tower_range and dist < min_dist:
                    min_dist = dist
                    target = zombie
            
            if target:
                tower['cooldown'] = self.TOWER_BASIC_COOLDOWN if is_basic else self.TOWER_ADVANCED_COOLDOWN
                damage = self.TOWER_BASIC_DAMAGE if is_basic else self.TOWER_ADVANCED_DAMAGE
                self.projectiles.append({
                    'pos': list(tower['px_pos']),
                    'target': target,
                    'damage': damage,
                    'speed': 8
                })
                # SFX: tower_shoot.wav

        # Update projectiles
        killed_count = 0
        self.projectiles[:] = [p for p in self.projectiles if p['target'] in self.zombies]
        for proj in self.projectiles:
            target_pos = proj['target']['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['speed']:
                proj['target']['health'] -= proj['damage']
                self._add_particles(proj['pos'], 5, (255, 200, 0), 2, 10)
                # SFX: zombie_hit.wav
                if proj['target']['health'] <= 0:
                    killed_count += 1
                    self._add_particles(proj['target']['pos'], 20, self.COLOR_ZOMBIE, 8, 20)
                    # SFX: zombie_death.wav
                    self.zombies.remove(proj['target'])
                self.projectiles.remove(proj)
            else:
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']
        
        return killed_count

    def _update_zombies(self):
        ended_count = 0
        for z in self.zombies:
            if z['waypoint_idx'] >= len(self.path_waypoints_px):
                self.zombies.remove(z)
                ended_count += 1
                continue

            target_wp = self.path_waypoints_px[z['waypoint_idx']]
            dx = target_wp[0] - z['pos'][0]
            dy = target_wp[1] - z['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < z['speed']:
                z['pos'] = list(target_wp)
                z['waypoint_idx'] += 1
            else:
                z['pos'][0] += (dx / dist) * z['speed']
                z['pos'][1] += (dy / dist) * z['speed']
        return ended_count

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)
        self.particles[:] = [p for p in self.particles if p['life'] > 0]

    def _add_particles(self, pos, count, color, max_radius, max_life):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'radius': self.np_random.random() * max_radius,
                'life': self.np_random.integers(max_life // 2, max_life)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw path and end zone
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints_px, self.GRID_SIZE)
        end_zone_rect = pygame.Rect(self.path_waypoints_px[-1][0] - self.GRID_SIZE//2, self.path_waypoints_px[-1][1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_END_ZONE, end_zone_rect)

        # Draw towers
        for tower in self.towers:
            color = self.COLOR_TOWER_BASIC if tower['type'] == 'basic' else self.COLOR_TOWER_ADVANCED
            pygame.draw.circle(self.screen, color, tower['px_pos'], self.GRID_SIZE // 3)
            if tower['cooldown'] > 0:
                pygame.draw.circle(self.screen, (0,0,0,100), tower['px_pos'], self.GRID_SIZE // 3)


        # Draw zombies
        for z in self.zombies:
            pos_int = (int(z['pos'][0]), int(z['pos'][1]))
            pulse = (math.sin(self.steps * 0.2) + 1) / 4 + 0.75 # 0.75 to 1.25
            radius = int(self.GRID_SIZE // 4 * pulse)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ZOMBIE)
            # Health bar
            health_ratio = z['health'] / z['max_health']
            bar_w = self.GRID_SIZE // 2
            bar_h = 4
            bar_x = pos_int[0] - bar_w//2
            bar_y = pos_int[1] - radius - bar_h - 2
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, int(bar_w * health_ratio), bar_h))

        # Draw projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p['pos'][0]), int(p['pos'][1])), 3)
            
        # Draw particles
        for p in self.particles:
            if p['radius'] > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Resources
        res_text = self.font_ui.render(f"Resources: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"Wave: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width()//2, 10))

        # Phase message
        msg = ""
        if self.game_phase == "placement" and self.wave_number <= self.TOTAL_WAVES:
            msg = f"Wave {self.wave_number} starts in: {math.ceil(self.phase_timer / 30)}"
        
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
        
        if msg:
            msg_surf = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "zombies_remaining": len(self.zombies) + len(self.zombies_to_spawn),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    done = False
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    print(env.game_description)

    while not done:
        # Map pygame keys to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over. Final Score: {info['score']}, Wave: {info['wave']}")
    env.close()