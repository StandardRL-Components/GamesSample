
# Generated: 2025-08-28T02:12:13.011359
# Source Brief: brief_04376.md
# Brief Index: 4376

        
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
        "Controls: Arrow keys to move. Survive as long as you can."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a deadly field of pulsating spikes. Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30  # Standard frame rate for smooth visuals
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # Player
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5

        # Spikes
        self.NUM_SPIKES = 25
        self.SPIKE_SIZE = 24 # A bit larger for better visibility
        self.SPIKE_PULSE_SPEED = 4

        # Colors
        self.COLOR_BG = (15, 15, 25) # Dark blue/black
        self.COLOR_PLAYER = (50, 255, 50) # Bright green
        self.COLOR_SPIKE_MAX = (255, 50, 50) # Bright red
        self.COLOR_SPIKE_MIN_BRIGHTNESS = 100
        self.COLOR_UI = (220, 220, 220) # Off-white

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Randomness ---
        self.np_random = None

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.steps_remaining = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.spikes = []

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Initialize game state
        self.steps = 0
        self.steps_remaining = self.MAX_STEPS
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = pygame.Vector2(
            self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        )

        # Spike state
        self.spikes = []
        min_dist_from_start = 100
        for _ in range(self.NUM_SPIKES):
            while True:
                pos = pygame.Vector2(
                    self.np_random.integers(self.SPIKE_SIZE, self.SCREEN_WIDTH - self.SPIKE_SIZE),
                    self.np_random.integers(self.SPIKE_SIZE, self.SCREEN_HEIGHT - self.SPIKE_SIZE),
                )
                if pos.distance_to(self.player_pos) > min_dist_from_start:
                    break
            
            self.spikes.append({
                "pos": pos,
                "brightness": self.np_random.integers(self.COLOR_SPIKE_MIN_BRIGHTNESS, 256),
                "pulse_dir": self.np_random.choice([-1, 1])
            })
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Game Logic ---
        self.steps += 1
        self.steps_remaining -= 1

        # 1. Unpack and handle actions
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # No function
        # shift_held = action[2] == 1  # No function
        self._handle_movement(movement)
        
        # 2. Update dynamic elements (spike pulsing)
        self._update_spikes()

        # 3. Check for termination conditions
        collided = self._check_collision()
        time_up = self.steps_remaining <= 0
        terminated = collided or time_up
        self.game_over = terminated

        # 4. Calculate reward
        reward = self._calculate_reward(collided, time_up)
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_movement(self, movement):
        """Updates player position based on the movement action."""
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        # movement == 0 is no-op

        # Clamp player position to screen boundaries
        self.player_pos.x = max(
            self.PLAYER_SIZE // 2, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE // 2)
        )
        self.player_pos.y = max(
            self.PLAYER_SIZE // 2, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_SIZE // 2)
        )

    def _update_spikes(self):
        """Updates the brightness of each spike to create a pulsing effect."""
        for spike in self.spikes:
            spike["brightness"] += spike["pulse_dir"] * self.SPIKE_PULSE_SPEED
            if not (self.COLOR_SPIKE_MIN_BRIGHTNESS <= spike["brightness"] <= 255):
                spike["brightness"] = np.clip(spike["brightness"], self.COLOR_SPIKE_MIN_BRIGHTNESS, 255)
                spike["pulse_dir"] *= -1

    def _check_collision(self):
        """Checks for collision between the player and any spike."""
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE // 2,
            self.player_pos.y - self.PLAYER_SIZE // 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        for spike in self.spikes:
            # Simple but effective AABB collision
            spike_rect = pygame.Rect(
                spike["pos"].x - self.SPIKE_SIZE // 2,
                spike["pos"].y - self.SPIKE_SIZE // 2,
                self.SPIKE_SIZE,
                self.SPIKE_SIZE,
            )
            if player_rect.colliderect(spike_rect):
                # Placeholder for sound effect
                # play_sound("collision.wav")
                return True
        return False
    
    def _calculate_reward(self, collided, time_up):
        """Calculates the reward for the current step."""
        if collided:
            return 0  # End of episode, no reward/penalty

        if time_up:
            # Placeholder for sound effect
            # play_sound("win.wav")
            return 100.0  # Big reward for winning

        return 0.01  # Small reward for surviving a step

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        """Renders the player and spikes."""
        # Render spikes
        for spike in self.spikes:
            color = (spike["brightness"], self.COLOR_SPIKE_MAX[1], self.COLOR_SPIKE_MAX[2])
            p1 = (int(spike["pos"].x), int(spike["pos"].y - self.SPIKE_SIZE // 2))
            p2 = (int(spike["pos"].x - self.SPIKE_SIZE // 2), int(spike["pos"].y + self.SPIKE_SIZE // 2))
            p3 = (int(spike["pos"].x + self.SPIKE_SIZE // 2), int(spike["pos"].y + self.SPIKE_SIZE // 2))
            # Use anti-aliased drawing for smoother triangles
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], color)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], color)

        # Render player
        player_rect = pygame.Rect(
            int(self.player_pos.x - self.PLAYER_SIZE // 2),
            int(self.player_pos.y - self.PLAYER_SIZE // 2),
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        # Draw a slight glow/outline for the player
        glow_rect = player_rect.inflate(4, 4)
        glow_color = (*self.COLOR_PLAYER, 100) # RGBA
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
    
    def _render_ui(self):
        """Renders the UI elements like the timer."""
        time_left = max(0, self.steps_remaining / self.FPS)
        time_text = f"TIME: {time_left:.2f}"
        text_surface = self.font_small.render(time_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, 10))
        
        # Display win/loss message
        if self.game_over:
            if self.steps_remaining <= 0:
                message = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                message = "GAME OVER"
                color = self.COLOR_SPIKE_MAX
            
            message_surface = self.font_large.render(message, True, color)
            pos = (
                self.SCREEN_WIDTH // 2 - message_surface.get_width() // 2,
                self.SCREEN_HEIGHT // 2 - message_surface.get_height() // 2,
            )
            self.screen.blit(message_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, self.steps_remaining / self.FPS)
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    # Set up Pygame for human interaction
    pygame.init()
    pygame.display.set_caption("Spike Field Survival")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    # Create the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                terminated = False

        if terminated:
            # If the game is over, just display the final frame
            # The user can press 'R' to reset
            pass
        else:
            # --- Action Mapping for Human Player ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        # Need to transpose it back for pygame's display format (width, height, channels)
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()