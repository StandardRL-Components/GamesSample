import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:00:33.405087
# Source Brief: brief_00147.md
# Brief Index: 147
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Stack blocks as high as you can with a wobbly crane. Place blocks carefully to keep the tower from toppling over."
    user_guide = "Use ←→ arrow keys to move the crane. Press space to drop a block."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_GROUND = (87, 56, 34)
        self.COLOR_BLOCK = (180, 180, 190)
        self.COLOR_BLOCK_OUTLINE = (100, 100, 110)
        self.COLOR_CRANE = (255, 220, 0)
        self.COLOR_CRANE_BEAM = (255, 220, 0, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_DIAL_BG = (255, 255, 255, 20)
        self.COLOR_DIAL_SAFE = (0, 255, 128)
        self.COLOR_DIAL_WARN = (255, 180, 0)
        self.COLOR_DIAL_DANGER = (255, 50, 50)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        # Game parameters
        self.MAX_TIME = 30.0
        self.WIN_HEIGHT = 15
        self.MAX_TILT_DEGREES = 20
        self.CRANE_SPEED = 8
        self.BLOCK_WIDTH = 60
        self.BLOCK_HEIGHT = 12
        self.GROUND_HEIGHT = 20
        self.TILT_SENSITIVITY = 0.4
        self.RANDOM_TILT_CHANCE = 0.25
        self.RANDOM_TILT_MAGNITUDE = 1.5
        self.MAX_EPISODE_STEPS = 1000

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = 0.0
        self.tower_blocks_offsets = []
        self.tower_height = 0
        self.tower_com = 0.0
        self.tower_tilt = 0.0
        self.crane_x = 0
        self.previous_space_state = False
        self.particles = []
        self.win_message = ""
        
        # Initialize state
        # self.reset() # reset is called by the test harness
        
        # Validate implementation
        # self.validate_implementation() # this runs step, which is not ideal in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = self.MAX_TIME
        self.tower_blocks_offsets = []
        self.tower_height = 0
        self.tower_com = 0.0
        self.tower_tilt = 0.0
        self.crane_x = self.WIDTH / 2
        self.previous_space_state = False
        self.particles = []
        self.win_message = ""
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        self.steps += 1
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle player input
        self._handle_input(movement, space_held)
        
        # 2. Update game state
        reward += self._update_physics(space_held)
        
        # 3. Check for termination
        terminated, terminal_reward = self._check_termination()
        self.game_over = terminated
        reward += terminal_reward
        
        # Update score with non-terminal rewards
        if not self.game_over:
            self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move crane left/right
        if movement == 3:  # Left
            self.crane_x -= self.CRANE_SPEED
        elif movement == 4: # Right
            self.crane_x += self.CRANE_SPEED
        
        # Clamp crane position to be within the screen bounds
        half_block = self.BLOCK_WIDTH / 2
        self.crane_x = np.clip(self.crane_x, half_block, self.WIDTH - half_block)

    def _update_physics(self, space_held):
        reward = 0
        
        # Timer countdown
        self.game_timer -= 1 / self.FPS
        
        # Check for block drop (on space button press)
        if space_held and not self.previous_space_state:
            # SFX: Block place click
            drop_offset = self.crane_x - self.WIDTH / 2
            
            # Update center of mass
            if self.tower_height > 0:
                self.tower_com = (self.tower_com * self.tower_height + drop_offset) / (self.tower_height + 1)
            else:
                self.tower_com = drop_offset

            self.tower_blocks_offsets.append(drop_offset)
            self.tower_height += 1
            
            # Update tilt based on new center of mass
            self.tower_tilt = self.tower_com * self.TILT_SENSITIVITY
            
            # Add random "blindfolded" tilt increase
            if self.np_random.random() < self.RANDOM_TILT_CHANCE:
                # SFX: Wobble sound
                random_wobble = self.np_random.uniform(-self.RANDOM_TILT_MAGNITUDE, self.RANDOM_TILT_MAGNITUDE)
                self.tower_tilt += random_wobble
            
            # Spawn particles on successful placement
            block_y_pos = self.HEIGHT - self.GROUND_HEIGHT - (self.tower_height - 1) * self.BLOCK_HEIGHT
            self._spawn_particles((self.crane_x, block_y_pos), 20, self.COLOR_BLOCK)
            
            # Continuous reward for placing a block
            reward += 0.1
            self.score += 0.1

        self.previous_space_state = space_held
        return reward

    def _check_termination(self):
        # Check for tower collapse
        if abs(self.tower_tilt) > self.MAX_TILT_DEGREES:
            # SFX: Tower collapse crash
            self.win_message = "TOWER COLLAPSED!"
            self._spawn_particles((self.WIDTH/2, self.HEIGHT/2), 100, self.COLOR_DIAL_DANGER)
            return True, -100
        
        # Check for win condition
        if self.tower_height >= self.WIN_HEIGHT:
            # SFX: Victory fanfare
            self.win_message = "TOWER COMPLETE!"
            self.score += 105 # Add final bonus to score
            self._spawn_particles((self.WIDTH/2, self.HEIGHT/2), 200, self.COLOR_CRANE)
            return True, 105 # +100 for winning, +5 for height goal

        # Check for timeout
        if self.game_timer <= 0:
            # SFX: Timeout buzzer
            self.win_message = "TIME'S UP!"
            return True, -10
        
        # Check for max steps
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.win_message = "MAX STEPS REACHED"
            return True, -10 # Treat as a timeout

        return False, 0

    def _get_observation(self):
        # 1. Clear screen with background
        self._render_background()
        
        # 2. Render all game elements
        self._render_ground()
        self._render_tower()
        self._render_crane_and_effects()
        
        # 3. Render UI overlay
        self._render_ui()
        
        # 4. Render Game Over overlay if necessary
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.tower_height,
            "tilt": self.tower_tilt,
            "time_left": self.game_timer
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            # Interpolate color from top to bottom
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - self.GROUND_HEIGHT, self.WIDTH, self.GROUND_HEIGHT))

    def _render_tower(self):
        if self.tower_height == 0:
            return

        # Add a subtle visual wobble for game feel
        visual_wobble = math.sin(self.steps * 0.15) * (self.tower_height / self.WIN_HEIGHT)
        visual_tilt = self.tower_tilt + visual_wobble

        # Create a surface to draw the un-rotated tower onto
        tower_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pivot_x = self.WIDTH / 2
        pivot_y = self.HEIGHT - self.GROUND_HEIGHT
        
        for i, offset in enumerate(self.tower_blocks_offsets):
            block_rect = pygame.Rect(
                pivot_x + offset - self.BLOCK_WIDTH / 2,
                pivot_y - (i + 1) * self.BLOCK_HEIGHT,
                self.BLOCK_WIDTH,
                self.BLOCK_HEIGHT
            )
            pygame.draw.rect(tower_surf, self.COLOR_BLOCK, block_rect, border_radius=2)
            pygame.draw.rect(tower_surf, self.COLOR_BLOCK_OUTLINE, block_rect, 1, border_radius=2)
        
        # Rotate the entire tower surface around the pivot point
        rotated_tower = pygame.transform.rotate(tower_surf, -visual_tilt) # Pygame rotation is counter-clockwise
        rot_rect = rotated_tower.get_rect(center=(pivot_x, pivot_y))
        
        self.screen.blit(rotated_tower, rot_rect)

    def _render_crane_and_effects(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['life'] * (255 / 30))))
                color = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life']/5), color)

        # Draw crane
        crane_y = self.HEIGHT - self.GROUND_HEIGHT - self.tower_height * self.BLOCK_HEIGHT - 30
        crane_y = max(50, crane_y)
        
        # Crane beam (visual guide)
        beam_rect = pygame.Rect(self.crane_x - 1, crane_y, 2, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_CRANE_BEAM, beam_rect)
        
        # Crane body
        pygame.draw.polygon(self.screen, self.COLOR_CRANE, [
            (int(self.crane_x), crane_y),
            (int(self.crane_x) - 10, crane_y - 10),
            (int(self.crane_x) + 10, crane_y - 10)
        ])
        pygame.gfxdraw.aacircle(self.screen, int(self.crane_x), crane_y - 15, 5, self.COLOR_CRANE)
        pygame.gfxdraw.filled_circle(self.screen, int(self.crane_x), crane_y - 15, 5, self.COLOR_CRANE)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, x, y, shadow_color, center=False):
            text_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = (x + 2, y + 2)
            else: text_rect.topleft = (x + 2, y + 2)
            self.screen.blit(text_surf, text_rect)
            
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = (x, y)
            else: text_rect.topleft = (x, y)
            self.screen.blit(text_surf, text_rect)

        # Draw UI text
        draw_text(f"SCORE: {int(self.score)}", self.font_small, self.COLOR_TEXT, 10, 10, self.COLOR_TEXT_SHADOW)
        draw_text(f"TIME: {max(0, self.game_timer):.1f}", self.font_small, self.COLOR_TEXT, self.WIDTH - 100, 10, self.COLOR_TEXT_SHADOW)
        draw_text(f"HEIGHT: {self.tower_height} / {self.WIN_HEIGHT}", self.font_small, self.COLOR_TEXT, 10, 35, self.COLOR_TEXT_SHADOW)
        
        # Draw tilt indicator dial
        dial_center = (self.WIDTH / 2, self.HEIGHT - 45)
        dial_radius = 60
        
        # Background arc
        pygame.gfxdraw.arc(self.screen, int(dial_center[0]), int(dial_center[1]), dial_radius, 200, 340, self.COLOR_DIAL_BG)

        # Danger zones
        pygame.gfxdraw.arc(self.screen, int(dial_center[0]), int(dial_center[1]), dial_radius-2, 200, 230, self.COLOR_DIAL_DANGER)
        pygame.gfxdraw.arc(self.screen, int(dial_center[0]), int(dial_center[1]), dial_radius-2, 310, 340, self.COLOR_DIAL_DANGER)
        pygame.gfxdraw.arc(self.screen, int(dial_center[0]), int(dial_center[1]), dial_radius-2, 230, 250, self.COLOR_DIAL_WARN)
        pygame.gfxdraw.arc(self.screen, int(dial_center[0]), int(dial_center[1]), dial_radius-2, 290, 310, self.COLOR_DIAL_WARN)
        pygame.gfxdraw.arc(self.screen, int(dial_center[0]), int(dial_center[1]), dial_radius-2, 250, 290, self.COLOR_DIAL_SAFE)
        
        # Needle
        tilt_ratio = self.tower_tilt / self.MAX_TILT_DEGREES
        angle = math.radians(180 * -tilt_ratio) # Map tilt to angle
        needle_end = (
            dial_center[0] + (dial_radius - 5) * math.sin(angle),
            dial_center[1] - (dial_radius - 5) * math.cos(angle)
        )
        pygame.draw.line(self.screen, self.COLOR_TEXT, dial_center, needle_end, 2)
        pygame.gfxdraw.filled_circle(self.screen, int(dial_center[0]), int(dial_center[1]), 4, self.COLOR_TEXT)
        
        # Tilt value text
        draw_text(f"{self.tower_tilt:.1f}°", self.font_small, self.COLOR_TEXT, dial_center[0], dial_center[1] + 15, self.COLOR_TEXT_SHADOW, center=True)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))
        
        # Helper to draw text with shadow
        def draw_text(text, font, color, x, y, shadow_color, center=False):
            text_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = (x + 3, y + 3)
            else: text_rect.topleft = (x + 3, y + 3)
            self.screen.blit(text_surf, text_rect)
            
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = (x, y)
            else: text_rect.topleft = (x, y)
            self.screen.blit(text_surf, text_rect)
            
        draw_text(self.win_message, self.font_large, self.COLOR_TEXT, self.WIDTH / 2, self.HEIGHT / 2, self.COLOR_TEXT_SHADOW, center=True)

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'color': color})

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
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Wobbly Tower")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_reward = 0
            continue

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Wait a bit before closing or allow reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        done = True # to exit the outer loop
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        done = False
                        wait_for_reset = False
                # Re-render the final screen while waiting
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(10) # Don't burn CPU while waiting

    env.close()