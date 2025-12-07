
# Generated: 2025-08-28T01:35:40.095688
# Source Brief: brief_04162.md
# Brief Index: 4162

        
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
        "Controls: ←→ to move, ↑ to jump. Reach the green exit pipe!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated pipe-themed platformer, leaping across gaps and dodging spinning hazards to reach the exit as quickly as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 2500
        self.TIME_LIMIT_SECONDS = 45

        # Player physics
        self.GRAVITY = 0.8
        self.PLAYER_JUMP_STRENGTH = -12
        self.PLAYER_HORIZ_ACCEL = 1.2
        self.PLAYER_MAX_SPEED = 7
        self.PLAYER_FRICTION = 0.85

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BG_PIPES = (30, 45, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 220, 255)
        self.COLOR_PLATFORM = (100, 110, 120)
        self.COLOR_PLATFORM_SHADOW = (60, 70, 80)
        self.COLOR_HAZARD = (255, 80, 80)
        self.COLOR_HAZARD_CENTER = (150, 40, 40)
        self.COLOR_EXIT = (80, 220, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.match_font('consolas', 'courier new', 'monospace'), 20)
            self.font_big = pygame.font.Font(pygame.font.match_font('consolas', 'courier new', 'monospace'), 48)
        except:
            self.font_ui = pygame.font.SysFont('monospace', 20)
            self.font_big = pygame.font.SysFont('monospace', 48)
        
        # Initialize state variables
        self.player_rect = None
        self.player_vel = None
        self.platforms = []
        self.hazards = []
        self.exit_rect = None
        self.background_pipes = []
        self.on_ground = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.max_y_reached = self.SCREEN_HEIGHT
        self.hazard_rot_speed = 0.0
        self.jump_squash = 0
        
        # Generate static background elements once
        self._generate_background()
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.hazard_rot_speed = 2.0
        
        self._generate_level()

        start_platform = self.platforms[0]
        self.player_rect = pygame.Rect(
            start_platform.centerx - 10, start_platform.top - 20, 20, 20
        )
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.max_y_reached = self.player_rect.centery
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.01  # Small reward for surviving
        
        # --- Player Input ---
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_HORIZ_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_HORIZ_ACCEL
        
        if movement == 1 and self.on_ground: # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.jump_squash = 5 # Visual effect
            # sfx: jump

        # --- Physics and Movement Update ---
        # Apply friction and clamp horizontal speed
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, 15) # Terminal velocity

        # Move horizontally
        self.player_rect.x += self.player_vel.x
        
        # Horizontal collision with platforms (simple wall blocking)
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.x > 0: # Moving right
                    self.player_rect.right = plat.left
                    self.player_vel.x = 0
                elif self.player_vel.x < 0: # Moving left
                    self.player_rect.left = plat.right
                    self.player_vel.x = 0

        # Move vertically
        self.player_rect.y += self.player_vel.y
        self.on_ground = False
        
        # Vertical collision with platforms
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0: # Moving down
                    self.player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    self.on_ground = True
                    # sfx: land
                elif self.player_vel.y < 0: # Moving up
                    self.player_rect.top = plat.bottom
                    self.player_vel.y = 0

        # Keep player on screen horizontally
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)

        # --- Update Game State ---
        self.steps += 1
        self.time_left -= 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.hazard_rot_speed += 0.5

        # Check for new height bonus
        if self.player_rect.centery < self.max_y_reached:
            reward += 10
            self.max_y_reached = self.player_rect.centery

        # --- Termination Checks ---
        terminated = False
        if self.player_rect.top > self.SCREEN_HEIGHT: # Fell off
            reward -= 10 # Penalty for falling
            terminated = True
            # sfx: fall
        
        if self.player_rect.colliderect(self.exit_rect): # Reached exit
            reward += 100
            terminated = True
            # sfx: win
            
        for hazard in self.hazards:
            hazard_center = pygame.Vector2(hazard['pos'])
            if hazard_center.distance_to(self.player_rect.center) < hazard['size']:
                reward -= 5
                terminated = True
                # sfx: hurt
                break

        if self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _generate_level(self):
        self.platforms = []
        self.hazards = []
        
        # Start platform
        start_plat = pygame.Rect(self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT - 40, 100, 20)
        self.platforms.append(start_plat)
        
        last_plat = start_plat
        for i in range(15):
            w = self.np_random.integers(60, 150)
            
            # Ensure next platform is reachable
            max_dx = self.PLAYER_MAX_SPEED * 8 + (self.PLAYER_JUMP_STRENGTH**2)/(2*self.GRAVITY) # Heuristic
            dx = self.np_random.integers(-max_dx, max_dx)
            dy = self.np_random.integers(70, 110)

            x = last_plat.centerx + dx - w // 2
            y = last_plat.top - dy

            # Clamp to screen
            x = np.clip(x, 0, self.SCREEN_WIDTH - w)
            y = max(50, y) # Leave space for UI

            new_plat = pygame.Rect(x, y, w, 20)
            self.platforms.append(new_plat)
            last_plat = new_plat

            # Add hazards occasionally
            if i > 2 and self.np_random.random() < 0.3:
                hazard_pos = (new_plat.centerx + self.np_random.integers(-w//2, w//2), new_plat.top - 20)
                self.hazards.append({'pos': hazard_pos, 'angle': self.np_random.random() * 360, 'size': 15})

        # Place exit on the last platform
        self.exit_rect = pygame.Rect(last_plat.centerx - 15, last_plat.top - 30, 30, 30)

    def _generate_background(self):
        self.background_pipes = []
        for _ in range(20):
            is_vertical = self.np_random.random() > 0.5
            if is_vertical:
                w, h = 30, self.np_random.integers(100, 300)
                x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
                y = self.np_random.integers(0, self.SCREEN_HEIGHT - h)
            else:
                w, h = self.np_random.integers(100, 400), 30
                x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
                y = self.np_random.integers(0, self.SCREEN_HEIGHT - h)
            self.background_pipes.append(pygame.Rect(x, y, w, h))

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background pipes
        for pipe in self.background_pipes:
            pygame.draw.rect(self.screen, self.COLOR_BG_PIPES, pipe)

        # Platforms
        for plat in self.platforms:
            shadow_rect = plat.copy()
            shadow_rect.y += 4
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, shadow_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)

        # Exit
        shadow_rect = self.exit_rect.copy()
        shadow_rect.y += 3
        pygame.draw.rect(self.screen, (30,100,30), shadow_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, border_radius=5)
        
        # Hazards
        for hazard in self.hazards:
            hazard['angle'] = (hazard['angle'] + self.hazard_rot_speed) % 360
            angle_rad = math.radians(hazard['angle'])
            size = hazard['size']
            center = hazard['pos']
            
            for i in range(3):
                rot = i * (2 * math.pi / 3)
                start_pos = (
                    int(center[0] + (size * 0.2) * math.cos(angle_rad + rot)),
                    int(center[1] + (size * 0.2) * math.sin(angle_rad + rot))
                )
                end_pos = (
                    int(center[0] + size * math.cos(angle_rad + rot)),
                    int(center[1] + size * math.sin(angle_rad + rot))
                )
                pygame.draw.aaline(self.screen, self.COLOR_HAZARD, start_pos, end_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), 5, self.COLOR_HAZARD_CENTER)
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), 5, self.COLOR_HAZARD)

        # Player
        if self.jump_squash > 0:
            squash_factor = self.jump_squash / 5.0
            squashed_rect = self.player_rect.copy()
            squashed_rect.height = int(20 * (1 - 0.4 * squash_factor))
            squashed_rect.width = int(20 * (1 + 0.4 * squash_factor))
            squashed_rect.center = self.player_rect.center
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, squashed_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, squashed_rect, 2, border_radius=4)
            self.jump_squash -= 1
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, self.player_rect.inflate(4, 4), 2, border_radius=6)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=4)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Score
        score_str = f"SCORE: {int(self.score)}"
        draw_text(score_str, self.font_ui, self.COLOR_TEXT, (10, 10))

        # Timer
        time_str = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        time_surf = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        draw_text(time_str, self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        if self.game_over:
            if self.player_rect.colliderect(self.exit_rect):
                msg = "YOU WIN!"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_HAZARD
            
            msg_surf = self.font_big.render(msg, True, color)
            msg_pos = (self.SCREEN_WIDTH//2 - msg_surf.get_width()//2, self.SCREEN_HEIGHT//2 - msg_surf.get_height()//2)
            
            shadow_surf = self.font_big.render(msg, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (msg_pos[0]+3, msg_pos[1]+3))
            self.screen.blit(msg_surf, msg_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_pos": (self.player_rect.x, self.player_rect.y),
            "player_vel": (self.player_vel.x, self.player_vel.y)
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


if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pipe Jumper")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Action mapping for human play
        # MultiDiscrete([5, 2, 2])
        # - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        # - actions[1]: Space button (0=released, 1=held)
        # - actions[2]: Shift button (0=released, 1=held)
        
        movement = 0 # none
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()