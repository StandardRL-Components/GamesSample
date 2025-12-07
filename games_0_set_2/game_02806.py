
# Generated: 2025-08-27T21:30:11.940505
# Source Brief: brief_02806.md
# Brief Index: 2806

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    An arcade-style side-scrolling platformer where the player controls a robot.
    The goal is to navigate through three progressively harder stages, collecting
    coins and avoiding obstacles, before a timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Collect yellow coins and reach the green finish line."
    )

    # Short, user-facing description of the game
    game_description = (
        "Guide a running robot through obstacle-filled stages, collecting coins and racing against the clock to reach the finish line."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors (Bright for interactive, Dark for background)
    COLOR_BG_DEEP = (15, 25, 35)
    COLOR_BG_MID = (25, 40, 55)
    COLOR_BG_NEAR = (40, 60, 80)
    COLOR_PLATFORM = (80, 100, 120)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_COIN = (255, 220, 0)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_FINISH_LINE = (50, 255, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Physics
    FPS = 30
    GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -14
    PLAYER_SPEED = 6
    PLAYER_FRICTION = 0.8

    # Game Rules
    MAX_LIVES = 3
    NUM_STAGES = 3
    STAGE_TIME_LIMIT_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.current_stage = 0
        self.stage_timer = 0
        self.player = {}
        self.platforms = []
        self.obstacles = []
        self.coins = []
        self.finish_line = None
        self.particles = []
        self.camera_x = 0
        self.stage_width = 0
        self.stage_height = self.SCREEN_HEIGHT
        self.last_damage_time = -1000 # To create invulnerability frames
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.lives = self.MAX_LIVES
        self.current_stage = 1
        
        self._setup_stage(self.current_stage)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        """Generates the layout for the current stage."""
        self.stage_width = self.SCREEN_WIDTH * (3 + stage_num)
        self.stage_timer = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self.camera_x = 0
        self.particles.clear()
        
        # Player setup
        self.player = {
            'rect': pygame.Rect(100, self.stage_height - 100, 24, 32),
            'vx': 0, 'vy': 0,
            'on_ground': False
        }

        # Clear old entities
        self.platforms = []
        self.obstacles = []
        self.coins = []
        
        # Ground platform
        ground = pygame.Rect(0, self.stage_height - 40, self.stage_width, 40)
        self.platforms.append(ground)

        # Procedural generation based on stage number
        obstacle_density = 0.1 + (stage_num * 0.05)
        obstacle_speed = 0.5 + (stage_num * 0.5)
        platform_gap_chance = 0.2 + (stage_num * 0.05)

        x = 400
        while x < self.stage_width - 400:
            # Platform gaps
            if self.np_random.random() < platform_gap_chance:
                x += self.np_random.integers(100, 200)
                # Add a floating platform over the gap
                plat_y = self.stage_height - self.np_random.integers(100, 150)
                plat_w = self.np_random.integers(80, 150)
                self.platforms.append(pygame.Rect(x, plat_y, plat_w, 20))
                # Add coins on the platform
                for i in range(self.np_random.integers(1, 4)):
                    self.coins.append(pygame.Rect(x + i*30 + 10, plat_y - 30, 16, 16))
                x += plat_w + self.np_random.integers(50, 100)
                continue

            # Obstacles
            if self.np_random.random() < obstacle_density:
                is_moving = self.np_random.random() < (stage_num * 0.2)
                obs_h = self.np_random.integers(20, 50)
                obstacle = {
                    'rect': pygame.Rect(x, self.stage_height - 40 - obs_h, 20, obs_h),
                    'moving': is_moving,
                    'speed': obstacle_speed if is_moving else 0,
                    'direction': 1 if self.np_random.random() < 0.5 else -1,
                    'range': (x - 50, x + 50)
                }
                self.obstacles.append(obstacle)
            
            # Coins
            if self.np_random.random() < 0.2:
                for i in range(self.np_random.integers(2, 6)):
                     self.coins.append(pygame.Rect(x + i*25, self.stage_height - 80, 16, 16))

            x += self.np_random.integers(150, 300)

        # Finish line
        self.finish_line = pygame.Rect(self.stage_width - 150, self.stage_height - 140, 20, 100)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player['vx'] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player['vx'] = self.PLAYER_SPEED
        else:
            self.player['vx'] *= self.PLAYER_FRICTION

        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vy'] = self.PLAYER_JUMP_STRENGTH
            self.player['on_ground'] = False
            self._create_particles(self.player['rect'].midbottom, count=10, color=(200,200,200), life=10, speed_range=(1,3))
            # Sound: Jump

        # --- Physics & Game Logic Update ---
        reward = 0.01 # Small reward for surviving
        self.steps += 1
        self.stage_timer -= 1
        
        # Update player physics
        self.player['vy'] += self.GRAVITY
        self.player['rect'].x += int(self.player['vx'])
        self.player['rect'].y += int(self.player['vy'])
        self.player['on_ground'] = False

        # Keep player within horizontal world bounds
        self.player['rect'].left = max(0, self.player['rect'].left)
        self.player['rect'].right = min(self.stage_width, self.player['rect'].right)

        # Platform collisions
        for plat in self.platforms:
            if self.player['rect'].colliderect(plat) and self.player['vy'] > 0:
                # Check if player was above the platform in the previous frame
                if self.player['rect'].bottom - self.player['vy'] <= plat.top:
                    self.player['rect'].bottom = plat.top
                    self.player['vy'] = 0
                    if not self.player['on_ground']:
                        self._create_particles(self.player['rect'].midbottom, count=5, color=(150,150,150), life=8, speed_range=(0.5,2))
                    self.player['on_ground'] = True

        # Update moving obstacles
        for obs in self.obstacles:
            if obs['moving']:
                obs['rect'].x += obs['speed'] * obs['direction']
                if obs['rect'].x < obs['range'][0] or obs['rect'].x > obs['range'][1]:
                    obs['direction'] *= -1

        # --- Event Handling ---
        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player['rect'].colliderect(coin):
                collected_coins.append(coin)
                self.score += 10
                reward += 1
                self._create_particles(coin.center, count=15, color=self.COLOR_COIN, life=20, speed_range=(1,4))
                # Sound: Coin collect
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Obstacle collision (with invulnerability frames)
        is_invulnerable = (self.steps - self.last_damage_time) < self.FPS * 1.5
        if not is_invulnerable:
            for obs in self.obstacles:
                if self.player['rect'].colliderect(obs['rect']):
                    self.lives -= 1
                    reward -= 5
                    self.last_damage_time = self.steps
                    self._create_particles(self.player['rect'].center, count=30, color=self.COLOR_OBSTACLE, life=30, speed_range=(2,6))
                    # Sound: Damage
                    # Knockback
                    self.player['vy'] = -5
                    self.player['vx'] = -self.player['vx'] * 0.5
                    break
        
        # --- Termination & Progression ---
        terminated = False
        
        # Finish line
        if self.player['rect'].colliderect(self.finish_line):
            self.score += 100
            reward += 10
            self.current_stage += 1
            if self.current_stage > self.NUM_STAGES:
                self.win_state = True
                terminated = True
                reward = 100 # Large win reward
            else:
                self._setup_stage(self.current_stage)
                # Sound: Stage Clear

        # Check loss conditions
        if self.lives <= 0 or self.stage_timer <= 0:
            terminated = True
            reward = -100 # Large loss penalty
        
        self.game_over = terminated

        # Update camera to follow player
        self.camera_x = self.player['rect'].centerx - self.SCREEN_WIDTH / 2
        self.camera_x = max(0, min(self.camera_x, self.stage_width - self.SCREEN_WIDTH))

        # Update particles
        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG_DEEP)
        self._draw_parallax_layer(self.COLOR_BG_MID, 0.25, 300, 100)
        self._draw_parallax_layer(self.COLOR_BG_NEAR, 0.5, 150, 50)
        
        # --- Game World Elements (Camera-adjusted) ---
        for plat in self.platforms:
            self._draw_world_rect(plat, self.COLOR_PLATFORM)
        
        for obs in self.obstacles:
            self._draw_world_rect(obs['rect'], self.COLOR_OBSTACLE)

        # Coin animation
        coin_anim_phase = (self.steps % 30) / 30.0
        coin_width = int(abs(math.cos(coin_anim_phase * math.pi * 2)) * 16)
        for coin in self.coins:
            # Create a new rect for drawing to avoid modifying the original
            draw_rect = pygame.Rect(coin.x, coin.y, max(1, coin_width), coin.height)
            draw_rect.centerx = coin.centerx
            self._draw_world_rect(draw_rect, self.COLOR_COIN)

        self._draw_world_rect(self.finish_line, self.COLOR_FINISH_LINE)

        # --- Player ---
        self._draw_player()
        
        # --- Particles ---
        for p in self.particles:
            pos = (int(p['x'] - self.camera_x), int(p['y']))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

        # --- UI Overlay ---
        self._render_ui()

        # --- Game Over/Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            text_surf = self.font_big.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.current_stage,
            "time_left": self.stage_timer // self.FPS,
        }
        
    def _draw_player(self):
        """Draws the player with animations and effects."""
        is_invulnerable = (self.steps - self.last_damage_time) < self.FPS * 1.5
        if is_invulnerable and self.steps % 6 < 3:
            return # Blinking effect when invulnerable

        p_rect_cam = self.player['rect'].move(-self.camera_x, 0)

        # Body
        body_rect = pygame.Rect(p_rect_cam.x, p_rect_cam.y, p_rect_cam.width, p_rect_cam.height - 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        pygame.gfxdraw.aacircle(self.screen, body_rect.centerx, body_rect.centery-6, 5, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, body_rect.centerx, body_rect.centery-6, 5, self.COLOR_PLAYER_GLOW)

        # Legs animation
        if not self.player['on_ground']: # Jumping
            leg_y = body_rect.bottom
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx - 4, leg_y), (body_rect.centerx - 6, leg_y + 4), 3)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx + 4, leg_y), (body_rect.centerx + 6, leg_y + 4), 3)
        else: # Running
            anim_phase = (self.steps % 20) / 20.0
            leg_offset = int(math.sin(anim_phase * math.pi * 2) * 6)
            leg_y = body_rect.bottom
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx - 4, leg_y), (body_rect.centerx - 4 - leg_offset, leg_y + 4), 3)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx + 4, leg_y), (body_rect.centerx + 4 + leg_offset, leg_y + 4), 3)

    def _render_ui(self):
        """Draws the UI elements on top of the game."""
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        ui_surf.blit(score_text, (10, 8))
        
        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.current_stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.SCREEN_WIDTH/2, y=8)
        ui_surf.blit(stage_text, stage_rect)

        # Lives & Timer
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        timer_text = self.font_ui.render(f"TIME: {max(0, self.stage_timer // self.FPS)}", True, self.COLOR_TEXT)
        ui_surf.blit(lives_text, (self.SCREEN_WIDTH - 180, 8))
        ui_surf.blit(timer_text, (self.SCREEN_WIDTH - 90, 8))
        
        self.screen.blit(ui_surf, (0, 0))

    def _draw_parallax_layer(self, color, factor, size, offset_y):
        """Draws a parallax scrolling background layer."""
        for i in range(-1, int(self.SCREEN_WIDTH / size) + 2):
            x = (i * size) - (self.camera_x * factor) % size
            pygame.draw.rect(self.screen, color, (x, offset_y, size - 10, self.SCREEN_HEIGHT))

    def _draw_world_rect(self, rect, color, border_radius=0):
        """Helper to draw a rect adjusted for camera position."""
        cam_rect = rect.move(-self.camera_x, 0)
        if cam_rect.right > 0 and cam_rect.left < self.SCREEN_WIDTH:
             pygame.draw.rect(self.screen, color, cam_rect, border_radius=border_radius)

    def _create_particles(self, pos, count, color, life, speed_range):
        """Spawns a number of particles at a given position."""
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.uniform(speed_range[0], speed_range[1])
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * (p['life'] / p['max_life']))
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set a dummy video driver for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # --- Test with random actions ---
    print("--- Testing with random actions ---")
    obs, info = env.reset()
    print(f"Initial Info: {info}")
    terminated = False
    total_reward = 0
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}, Info: {info}, Reward: {reward:.2f}")
        if terminated:
            print(f"Episode terminated at step {i+1}. Final Info: {info}, Total Reward: {total_reward:.2f}")
            break
    
    # To visualize the game, you would need a rendering loop
    # This requires a non-dummy video driver
    # Example (won't run with the dummy driver setting):
    #
    # import os
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    #
    # env = GameEnv()
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Robot Runner")
    # clock = pygame.time.Clock()
    #
    # running = True
    # while running:
    #     # Map keyboard to actions
    #     keys = pygame.key.get_pressed()
    #     movement = 0 # none
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
    #     action = [movement, space, shift]
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     # Render the observation to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         pygame.time.wait(2000) # Pause before reset
    #         obs, info = env.reset()
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     clock.tick(GameEnv.FPS)
    #
    # env.close()