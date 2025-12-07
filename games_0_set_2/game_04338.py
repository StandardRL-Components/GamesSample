# Generated: 2025-08-28T02:06:05.725066
# Source Brief: brief_04338.md
# Brief Index: 4338

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press space to jump. Time your jumps to cross the gaps and reach the flag."

    # Must be a short, user-facing description of the game:
    game_description = "A minimalist platformer. Jump across procedurally generated gaps to reach the end of three stages as fast as possible."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    PLAYER_SIZE = 20
    GRAVITY = 0.6
    JUMP_STRENGTH = -12
    PLAYER_X_SPEED = 4
    STAGE_TIME_LIMIT_SECONDS = 60

    # --- Colors ---
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_PLAYER = (255, 255, 0)  # Yellow
    COLOR_PLAYER_OUTLINE = (200, 200, 0)
    COLOR_PLATFORM = (100, 100, 100)
    COLOR_PLATFORM_OUTLINE = (80, 80, 80)
    COLOR_FLAG_POLE = (101, 67, 33) # Brown
    COLOR_FLAG = (220, 20, 60) # Crimson
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_PARTICLE = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("sans-serif", 36, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.prev_space_held = None
        self.platforms = None
        self.flag_rect = None
        self.landed_platform_indices = None
        self.particles = None
        self.camera_offset = None
        self.squash_factor = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.stage = None
        self.time_left = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.stage = 1
        self.time_left = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        
        self.player_pos = np.array([100.0, 250.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = False
        self.prev_space_held = False

        self.platforms = []
        self.flag_rect = None
        self.landed_platform_indices = set()
        self.particles = []

        self.camera_offset = np.array([0.0, 0.0])
        
        # Squash and stretch animation state
        self.squash_factor = 0
        
        self._generate_stage()

        return self._get_observation(), self._get_info()

    def _generate_stage(self):
        """Generates platforms and the flag for the current stage."""
        self.platforms.clear()
        self.landed_platform_indices.clear()
        
        # Reset timer and player position for the new stage
        self.time_left = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self.player_pos = np.array([100.0, 250.0])
        self.player_vel = np.array([0.0, 0.0])
        
        # Starting platform
        start_platform = pygame.Rect(0, 300, 200, 100)
        self.platforms.append(start_platform)
        self.landed_platform_indices.add(0) # Player starts on this one

        current_x = start_platform.right
        last_y = start_platform.top

        if self.stage == 1:
            gap_min, gap_max, y_variation = 50, 80, 0
        elif self.stage == 2:
            gap_min, gap_max, y_variation = 100, 150, 0
        else: # Stage 3
            gap_min, gap_max, y_variation = 100, 150, 50

        for i in range(15): # Generate 15 platforms
            gap = self.np_random.integers(gap_min, gap_max + 1)
            current_x += gap
            
            plat_width = self.np_random.integers(100, 250)
            plat_height = 100
            
            y_delta = self.np_random.integers(-y_variation, y_variation + 1) if y_variation > 0 else 0
            plat_y = np.clip(last_y + y_delta, 150, 350)
            
            platform = pygame.Rect(current_x, plat_y, plat_width, plat_height)
            self.platforms.append(platform)
            
            current_x += plat_width
            last_y = plat_y

        # Place flag on the last platform
        last_platform = self.platforms[-1]
        self.flag_rect = pygame.Rect(last_platform.right - 40, last_platform.top - 60, 30, 60)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        terminated = False
        
        # --- Update Time and Survival Reward ---
        self.time_left -= 1
        reward += 0.1  # Survival reward

        if self.time_left <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True

        # --- Handle Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if space_held and not self.prev_space_held and self.is_grounded:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            self.squash_factor = 15 # Start jump-stretch animation
            # SFX: Jump sound
        self.prev_space_held = space_held

        # --- Update Physics ---
        # Set horizontal velocity based on the 'movement' action
        if movement == 4: # Right
            self.player_vel[0] = self.PLAYER_X_SPEED
        else: # No-op or other directions result in no horizontal movement
            self.player_vel[0] = 0.0

        if not self.is_grounded:
            self.player_vel[1] += self.GRAVITY
        
        self.player_pos += self.player_vel

        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        # --- Collision Detection ---
        self.is_grounded = False
        for i, plat in enumerate(self.platforms):
            if player_rect.colliderect(plat) and self.player_vel[1] >= 0:
                # Check if the player was above the platform in the previous frame
                prev_player_bottom = player_rect.bottom - self.player_vel[1]
                if prev_player_bottom <= plat.top:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.is_grounded = True
                    self.squash_factor = -15 # Start landing-squash animation
                    
                    if i not in self.landed_platform_indices:
                        reward += 5.0 # New platform bonus
                        self.landed_platform_indices.add(i)
                        # SFX: Land on new platform
                    
                    self._create_particles(player_rect.midbottom)
                    # SFX: Land sound
                    break
        
        # --- Update Game State ---
        # Fall off screen
        if self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            reward = -100.0
            terminated = True
            self.game_over = True
            # SFX: Fall/Fail sound
        
        # Reach flag
        if not terminated and self.flag_rect.colliderect(player_rect):
            reward += 100.0 # Stage clear bonus
            self.stage += 1
            if self.stage > 3:
                reward += 300.0 # Game win bonus
                terminated = True
                self.game_over = True
                # SFX: Win fanfare
            else:
                self._generate_stage()
                # SFX: Stage clear sound

        self.score += reward
        self._update_particles()
        
        self.steps += 1
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, pos):
        for _ in range(self.np_random.integers(5, 10)):
            vel = [self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-2, 0)]
            life = self.np_random.integers(15, 25)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Update Camera ---
        target_cam_x = self.player_pos[0] - 100
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.6
        self.camera_offset[0] += (target_cam_x - self.camera_offset[0]) * 0.1
        self.camera_offset[1] += (target_cam_y - self.camera_offset[1]) * 0.1

        # --- Render Platforms ---
        for plat in self.platforms:
            on_screen_plat = plat.move(-self.camera_offset)
            if on_screen_plat.right > 0 and on_screen_plat.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, on_screen_plat)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, on_screen_plat, 3)

        # --- Render Flag ---
        on_screen_flag = self.flag_rect.move(-self.camera_offset)
        pole_rect = pygame.Rect(on_screen_flag.left, on_screen_flag.top, 5, on_screen_flag.height)
        flag_poly = [
            (on_screen_flag.left + 5, on_screen_flag.top),
            (on_screen_flag.left + on_screen_flag.width, on_screen_flag.top + on_screen_flag.height / 3),
            (on_screen_flag.left + 5, on_screen_flag.top + on_screen_flag.height * 2/3)
        ]
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, pole_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_poly)
        
        # --- Render Particles ---
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_offset[0]), int(p['pos'][1] - self.camera_offset[1]))
            alpha = 255 * (p['life'] / p['max_life'])
            radius = int(3 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, pos, radius)

        # --- Render Player ---
        if self.squash_factor > 0: # Stretching for jump
            self.squash_factor -= 1
        elif self.squash_factor < 0: # Squashing for land
            self.squash_factor += 1
            
        squash_w = self.PLAYER_SIZE + self.squash_factor
        squash_h = self.PLAYER_SIZE - self.squash_factor
        
        player_screen_pos = self.player_pos - self.camera_offset
        player_rect = pygame.Rect(
            int(player_screen_pos[0] - (squash_w - self.PLAYER_SIZE)/2), 
            int(player_screen_pos[1] - (squash_h - self.PLAYER_SIZE)/2), 
            int(squash_w), 
            int(squash_h)
        )
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=3)

    def _render_ui(self):
        def draw_text(text, font, pos, color, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Stage
        stage_text = f"Stage: {self.stage}/3"
        draw_text(stage_text, self.font_small, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Time
        time_text = f"Time: {max(0, self.time_left // self.FPS)}"
        text_w = self.font_small.size(time_text)[0]
        draw_text(time_text, self.font_small, (self.SCREEN_WIDTH - text_w - 10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Score
        score_text = f"Score: {int(self.score)}"
        draw_text(score_text, self.font_small, (10, self.SCREEN_HEIGHT - 34), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over / Win Text
        if self.game_over:
            if self.stage > 3:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            text_w, text_h = self.font_large.size(msg)
            pos = ((self.SCREEN_WIDTH - text_w) / 2, (self.SCREEN_HEIGHT - text_h) / 2)
            draw_text(msg, self.font_large, pos, self.COLOR_FLAG, self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left_seconds": max(0, self.time_left // self.FPS),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Minimalist Platformer")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        # Unused in this game, but mapping them for completeness
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        movement = 0 # none
        if up: movement = 1
        elif down: movement = 2
        elif left: movement = 3
        elif right: movement = 4

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation 'obs' is the rendered game screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            print("--- Resetting Environment ---")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()