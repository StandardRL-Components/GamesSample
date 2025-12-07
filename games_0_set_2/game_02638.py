
# Generated: 2025-08-27T20:58:49.342089
# Source Brief: brief_02638.md
# Brief Index: 2638

        
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
        "Controls: ←→ or ↑↓ to select a tower slot. Space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000 # Increased for 3 waves

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_PATH = (60, 60, 80)
        self.COLOR_BASE = (50, 180, 50)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TOWER = (50, 150, 220)
        self.COLOR_PROJECTILE = (255, 220, 100)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_SLOT = (80, 80, 100)
        self.COLOR_SLOT_SELECTED = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)

        # Game configuration
        self.PATH_POINTS = [
            (-50, 200), (100, 200), (100, 100), (540, 100), (540, 300), (700, 300)
        ]
        self.TOWER_SLOTS = [
            (250, 150), (390, 150), (250, 250), (390, 250)
        ]
        self.WAVE_CONFIG = [
            {'count': 10, 'speed': 1.0, 'spawn_delay': 60, 'wave_delay': 90},
            {'count': 15, 'speed': 1.2, 'spawn_delay': 45, 'wave_delay': 90},
            {'count': 20, 'speed': 1.4, 'spawn_delay': 30, 'wave_delay': 0},
        ]
        self.TOTAL_WAVES = len(self.WAVE_CONFIG)
        
        # Tower properties
        self.TOWER_RANGE = 100
        self.TOWER_DAMAGE = 2
        self.TOWER_FIRE_RATE = 15 # frames (0.5s at 30fps)
        self.PROJECTILE_SPEED = 10

        # Enemy properties
        self.ENEMY_HEALTH = 10
        self.ENEMY_RADIUS = 8
        self.BASE_POS = (640, 300)
        self.BASE_RADIUS = 20

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.placed_tower_slots = set()
        self.selected_slot_idx = 0
        self.last_space_held = False

        self.current_wave_idx = -1
        self.wave_timer = self.WAVE_CONFIG[0]['wave_delay']
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack and handle player actions
            self._handle_actions(action)

            # Update game state
            self._update_waves()
            reward += self._update_towers()
            reward += self._update_projectiles()
            
            lose_condition, kill_reward = self._update_enemies()
            reward += kill_reward
            if lose_condition:
                self.game_over = True
                reward -= 100

            self._update_particles()
            
            # Check for win condition
            if not self.enemies and self.enemies_to_spawn == 0 and self.current_wave_idx == self.TOTAL_WAVES - 1:
                self.win = True
                self.game_over = True
                reward += 100

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Cycle through tower slots
        if movement in [1, 3]: # Up or Left
            self.selected_slot_idx = (self.selected_slot_idx - 1) % len(self.TOWER_SLOTS)
        elif movement in [2, 4]: # Down or Right
            self.selected_slot_idx = (self.selected_slot_idx + 1) % len(self.TOWER_SLOTS)

        # Place tower on space press
        if space_held and not self.last_space_held:
            if self.selected_slot_idx not in self.placed_tower_slots:
                # sfx: build_tower.wav
                pos = self.TOWER_SLOTS[self.selected_slot_idx]
                self.towers.append({'pos': pos, 'cooldown': 0, 'target': None})
                self.placed_tower_slots.add(self.selected_slot_idx)
        
        self.last_space_held = space_held

    def _start_next_wave(self):
        self.current_wave_idx += 1
        if self.current_wave_idx < self.TOTAL_WAVES:
            wave_cfg = self.WAVE_CONFIG[self.current_wave_idx]
            self.enemies_to_spawn = wave_cfg['count']
            self.spawn_timer = 0
            return 50 # Reward for surviving a wave
        return 0

    def _update_waves(self):
        # Check for wave completion and start next
        if not self.enemies and self.enemies_to_spawn == 0 and self.current_wave_idx < self.TOTAL_WAVES - 1:
            if self.wave_timer > 0:
                self.wave_timer -= 1
            else:
                self._start_next_wave()
                wave_cfg = self.WAVE_CONFIG[self.current_wave_idx]
                self.wave_timer = wave_cfg.get('wave_delay', 90)

        # Spawn enemies for the current wave
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                wave_cfg = self.WAVE_CONFIG[self.current_wave_idx]
                self.enemies_to_spawn -= 1
                self.spawn_timer = wave_cfg['spawn_delay']
                
                start_pos = self.PATH_POINTS[0]
                self.enemies.append({
                    'pos': list(start_pos),
                    'health': self.ENEMY_HEALTH,
                    'speed': wave_cfg['speed'],
                    'path_idx': 1
                })

    def _update_towers(self):
        hit_reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1

            # Retain target if still valid
            if tower['target'] and (tower['target'] not in self.enemies or \
               math.hypot(tower['pos'][0] - tower['target']['pos'][0], tower['pos'][1] - tower['target']['pos'][1]) > self.TOWER_RANGE):
                tower['target'] = None
            
            # Find new target if needed
            if not tower['target']:
                for enemy in self.enemies:
                    dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
                    if dist <= self.TOWER_RANGE:
                        tower['target'] = enemy
                        break
            
            # Fire projectile
            if tower['target'] and tower['cooldown'] == 0:
                # sfx: laser_shoot.wav
                tower['cooldown'] = self.TOWER_FIRE_RATE
                self.projectiles.append({
                    'start_pos': tower['pos'],
                    'target': tower['target']
                })
                hit_reward += 0.1 # Reward for firing and likely hitting
        return hit_reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            start_x, start_y = proj['start_pos']
            target_x, target_y = proj['target']['pos']
            
            # Move projectile towards target
            dx, dy = target_x - start_x, target_y - start_y
            dist = math.hypot(dx, dy)
            
            if dist < self.PROJECTILE_SPEED:
                # Hit target
                proj['target']['health'] -= self.TOWER_DAMAGE
                self._create_particles(proj['target']['pos'])
                self.projectiles.remove(proj)
                # sfx: enemy_hit.wav
            else:
                # Update projectile position
                move_x = (dx / dist) * self.PROJECTILE_SPEED
                move_y = (dy / dist) * self.PROJECTILE_SPEED
                proj['start_pos'] = (start_x + move_x, start_y + move_y)
        return 0

    def _update_enemies(self):
        kill_reward = 0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                # sfx: enemy_die.wav
                self.enemies.remove(enemy)
                self.score += 10
                kill_reward += 1
                continue

            if enemy['path_idx'] >= len(self.PATH_POINTS):
                continue

            target_pos = self.PATH_POINTS[enemy['path_idx']]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['path_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
            
            # Check if enemy reached base
            if math.hypot(enemy['pos'][0] - self.BASE_POS[0], enemy['pos'][1] - self.BASE_POS[1]) < self.BASE_RADIUS:
                # sfx: base_explode.wav
                self.enemies.remove(enemy)
                return True, kill_reward # Lose condition

        return False, kill_reward

    def _create_particles(self, pos, count=5):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': 10})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_POINTS, 30)
        
        # Draw base
        pygame.gfxdraw.filled_circle(self.screen, int(self.BASE_POS[0]), int(self.BASE_POS[1]), self.BASE_RADIUS, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.BASE_POS[0]), int(self.BASE_POS[1]), self.BASE_RADIUS, self.COLOR_BASE)

        # Draw tower slots
        for i, pos in enumerate(self.TOWER_SLOTS):
            color = self.COLOR_SLOT_SELECTED if i == self.selected_slot_idx else self.COLOR_SLOT
            radius = 20
            if i == self.selected_slot_idx:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 2, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BG)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Draw towers
        for tower in self.towers:
            x, y = int(tower['pos'][0]), int(tower['pos'][1])
            pygame.draw.rect(self.screen, self.COLOR_TOWER, (x-10, y-10, 20, 20))

        # Draw enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Draw projectiles
        for proj in self.projectiles:
            x, y = int(proj['start_pos'][0]), int(proj['start_pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 10)))
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, color, pos, max(0, int(p['life']/4)))


    def _render_ui(self):
        # Wave info
        wave_text = f"Wave: {self.current_wave_idx + 1} / {self.TOTAL_WAVES}"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Enemies remaining
        enemies_left = len(self.enemies) + self.enemies_to_spawn
        enemies_text = f"Enemies: {enemies_left}"
        text_surf = self.font_small.render(enemies_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"Score: {self.score}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT - 30))

        # Game Over / Win Text
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            pos = (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2)
            self.screen.blit(text_surf, pos)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave_idx + 1,
            "enemies_left": len(self.enemies) + self.enemies_to_spawn,
            "towers_placed": len(self.towers)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # --- Game Loop ---
    while not done:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it from the env and display it
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over. Final Info: {info}")
    env.close()