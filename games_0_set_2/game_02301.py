
# Generated: 2025-08-28T04:25:05.287537
# Source Brief: brief_02301.md
# Brief Index: 2301

        
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
        "Controls: Press SPACE to jump over obstacles in time with the beat."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm game. Navigate a neon road, jumping over obstacles "
        "to the rhythm of a dynamic soundtrack. Time your jumps perfectly to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_stage = pygame.font.Font(None, 48)
        self.font_gameover = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (16, 12, 44) # #100c2c
        self.COLOR_ROAD = (28, 22, 68)
        self.COLOR_ROAD_LINES = (100, 100, 220, 150)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 255, 255)
        self.OBSTACLE_COLORS = [(255, 0, 255), (255, 255, 0), (0, 255, 0)]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_FAIL = (255, 50, 50)

        # Game state variables
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        
        self.player_y = 0
        self.player_vy = 0
        self.is_jumping = False
        
        self.obstacles = []
        self.particles = []
        self.road_lines = []
        
        self.current_stage = 1
        self.missed_jumps = 0
        self.stage_timer = 0
        
        self.beat_timer = 0
        self.beat_pulse = 0
        
        # Physics and progression
        self.GROUND_Y = self.HEIGHT - 80
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        
        self.stage_configs = [
            {'speed': 10, 'interval': 60, 'theme': 0}, # 1 obs / 2s
            {'speed': 12, 'interval': 50, 'theme': 1}, # 1 obs / 1.66s
            {'speed': 14, 'interval': 43, 'theme': 2}, # 1 obs / 1.43s
        ]
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        
        self.current_stage = 1
        self._reset_stage()
        
        self.road_lines = []
        for _ in range(40):
            self.road_lines.append({
                'z': self.np_random.random() * 40,
                'x_offset': 0
            })
            
        return self._get_observation(), self._get_info()

    def _reset_stage(self):
        self.missed_jumps = 0
        self.stage_timer = 0
        self.beat_timer = 0
        
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_jumping = False
        
        self.obstacles = []
        self.particles = []

    def step(self, action):
        reward = 0
        terminated = False
        
        # Survival reward: +1 for every 0.1 seconds (3 steps)
        if self.steps % 3 == 0:
            reward += 0.1 # Scaled down from 1 to keep rewards in range

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            reward += self._update_obstacles()
            self._update_particles()
            self._update_road()
            
            # Update timers and beat
            self.stage_timer += 1
            self.beat_timer += 1
            
            # Check for stage/game end conditions
            if self.missed_jumps >= 3:
                self.game_over = True
                terminated = True
                reward = -50
            
            if self.stage_timer >= self.FPS * 60: # 60 second stage limit
                if self.current_stage < len(self.stage_configs):
                    self.current_stage += 1
                    self._reset_stage()
                    reward += 50
                else: # Beat the final stage
                    self.game_over = True
                    self.victory = True
                    terminated = True
                    reward += 100

        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        space_pressed = action[1] == 1
        if space_pressed and not self.is_jumping:
            self.is_jumping = True
            self.player_vy = self.JUMP_STRENGTH
            # SFX: Jump

    def _update_player(self):
        if self.is_jumping:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            self.is_jumping = False

    def _update_obstacles(self):
        reward = 0
        config = self.stage_configs[self.current_stage - 1]
        
        # Spawn new obstacles on the beat
        if self.beat_timer >= config['interval']:
            self.beat_timer = 0
            obstacle_height = self.np_random.integers(30, 60)
            self.obstacles.append({
                'rect': pygame.Rect(self.WIDTH, self.GROUND_Y - obstacle_height, 20, obstacle_height),
                'color': self.OBSTACLE_COLORS[config['theme']],
                'cleared': False
            })
            # SFX: Beat
        
        # Update obstacle positions and check for collisions/clears
        player_rect = self._get_player_rect()
        for obs in self.obstacles[:]:
            obs['rect'].x -= config['speed']
            
            # Check for successful jump
            if not obs['cleared'] and player_rect.left > obs['rect'].right:
                obs['cleared'] = True
                self.score += 10
                reward += 5
                self._spawn_particles(obs['rect'].midtop, obs['color'])
                # SFX: Success
            
            # Check for collision
            if player_rect.colliderect(obs['rect']):
                if not self.is_jumping or player_rect.bottom > obs['rect'].top + 10:
                    self.missed_jumps += 1
                    self.obstacles.remove(obs)
                    self._spawn_particles(player_rect.center, self.COLOR_FAIL, 50)
                    # SFX: Fail
                    continue
            
            if obs['rect'].right < 0:
                # If obstacle scrolled off screen without being jumped (and player on ground)
                if not obs['cleared'] and not self.is_jumping:
                    self.missed_jumps += 1
                    self._spawn_particles((10, self.HEIGHT // 2), self.COLOR_FAIL, 50)
                    # SFX: Fail
                self.obstacles.remove(obs)
                
        return reward

    def _spawn_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_road(self):
        config = self.stage_configs[self.current_stage - 1]
        for line in self.road_lines:
            line['z'] -= config['speed'] / 40.0
            if line['z'] <= 0:
                line['z'] = 40
                line['x_offset'] = (self.np_random.random() - 0.5) * 500

    def _get_player_rect(self):
        return pygame.Rect(self.WIDTH // 4, int(self.player_y) - 30, 20, 30)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Beat pulse effect
        config = self.stage_configs[self.current_stage - 1]
        beat_progress = self.beat_timer / config['interval']
        self.beat_pulse = math.sin(beat_progress * math.pi) ** 2
        
        # Background glow
        glow_radius = int(200 + 100 * self.beat_pulse)
        glow_color = self.OBSTACLE_COLORS[config['theme']]
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, (*glow_color, 20))
        self.screen.blit(temp_surf, (self.WIDTH // 2 - glow_radius, self.HEIGHT // 2 - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Road
        self._render_road()

        # Obstacles
        for obs in self.obstacles:
            # Glow
            glow_rect = obs['rect'].inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*obs['color'], 50), (0, 0, *glow_rect.size), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft)
            # Main rect
            pygame.draw.rect(self.screen, obs['color'], obs['rect'], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, obs['rect'].inflate(-4, -4), border_radius=3)
            pygame.draw.rect(self.screen, obs['color'], obs['rect'].inflate(-8, -8), border_radius=3)

        # Player
        player_rect = self._get_player_rect()
        # Glow
        glow_rect = player_rect.inflate(15, 15)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, int(80 + 40 * self.beat_pulse)), (0, 0, *glow_rect.size), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha))

    def _render_road(self):
        # Road surface
        road_poly = [
            (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y),
            (self.WIDTH, self.HEIGHT), (0, self.HEIGHT)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_ROAD, road_poly)

        # Perspective lines
        horizon_y = self.HEIGHT // 2 + 50
        for line in self.road_lines:
            z = line['z']
            # Perspective projection
            y = horizon_y + (self.HEIGHT - horizon_y) * (1 - (z / 40.0)**0.5)
            if y < self.GROUND_Y:
                continue
            
            scale = 1 - (z / 40.0)
            line_width = self.WIDTH * scale
            x = (self.WIDTH / 2) - (line_width / 2) + (line['x_offset'] * scale)
            
            alpha = 255 * (1 - (z / 40.0))
            if y < self.HEIGHT:
                pygame.draw.line(self.screen, (*self.COLOR_ROAD_LINES[:3], int(alpha)), (int(x), int(y)), (int(x + line_width), int(y)), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_ui.render(f"FAILS: {self.missed_jumps} / 3", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.WIDTH - miss_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_stage.render(f"STAGE {self.current_stage}", True, self.COLOR_TEXT)
        text_rect = stage_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 30))
        self.screen.blit(stage_text, text_rect)

        # Game Over / Victory message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else self.COLOR_FAIL
            
            end_text = self.font_gameover.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "missed_jumps": self.missed_jumps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over_screen = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        'up': False, 'down': False, 'left': False, 'right': False,
        'space': False, 'shift': False
    }

    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default no-op action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: keys_held['up'] = True
                if event.key == pygame.K_DOWN: keys_held['down'] = True
                if event.key == pygame.K_LEFT: keys_held['left'] = True
                if event.key == pygame.K_RIGHT: keys_held['right'] = True
                if event.key == pygame.K_SPACE: keys_held['space'] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held['shift'] = True
                if event.key == pygame.K_r and game_over_screen: # Reset on 'R'
                    obs, info = env.reset()
                    game_over_screen = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_held['up'] = False
                if event.key == pygame.K_DOWN: keys_held['down'] = False
                if event.key == pygame.K_LEFT: keys_held['left'] = False
                if event.key == pygame.K_RIGHT: keys_held['right'] = False
                if event.key == pygame.K_SPACE: keys_held['space'] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held['shift'] = False
        
        # Map held keys to MultiDiscrete action space
        if keys_held['up']: action[0] = 1
        elif keys_held['down']: action[0] = 2
        elif keys_held['left']: action[0] = 3
        elif keys_held['right']: action[0] = 4
        else: action[0] = 0
            
        if keys_held['space']: action[1] = 1
        if keys_held['shift']: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment's internal screen is not the display screen
        # We need to create a window and blit the observation to it.
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Rhythm Jumper")
        
        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            game_over_screen = True
            print(f"Game Over! Final Score: {info['score']}. Press 'R' to restart.")
        
        env.clock.tick(env.FPS)
        
    env.close()