
# Generated: 2025-08-28T04:15:04.244442
# Source Brief: brief_02259.md
# Brief Index: 2259

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. Avoid the red spikes."
    )

    game_description = (
        "A fast-paced arcade survival game. Maneuver your character to avoid deadly spikes "
        "and survive for 30 seconds to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 200, 200
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 30
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS + 10

        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 5
        self.NUM_SPIKES = 10
        self.SPIKE_SIZE = 12

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WORLD_BG = (40, 40, 50)
        self.COLOR_PLAYER = (50, 200, 255)
        self.COLOR_SPIKE = (255, 50, 50)
        self.COLOR_PARTICLE = (255, 80, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER = (200, 200, 200)
        self.COLOR_VICTORY = (100, 255, 100)
        self.COLOR_DEFEAT = (255, 100, 100)
        
        # --- World Position ---
        self.world_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.WORLD_WIDTH) / 2,
            (self.SCREEN_HEIGHT - self.WORLD_HEIGHT) / 2,
            self.WORLD_WIDTH,
            self.WORLD_HEIGHT
        )

        # --- Game State (initialized in reset) ---
        self.player_rect = None
        self.spikes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_remaining_frames = 0
        self.game_over = False
        self.victory = False

        # --- Initialize state variables ---
        self.reset()
        
        # --- Run validation check ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.time_remaining_frames = self.GAME_DURATION_SECONDS * self.FPS
        
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = self.world_rect.center
        
        self.spikes.clear()
        self.particles.clear()
        self._generate_spikes()

        return self._get_observation(), self._get_info()

    def _generate_spikes(self):
        player_safe_zone = self.player_rect.inflate(self.PLAYER_SIZE * 4, self.PLAYER_SIZE * 4)
        for _ in range(self.NUM_SPIKES):
            is_valid_pos = False
            while not is_valid_pos:
                x = self.np_random.integers(self.world_rect.left, self.world_rect.right - self.SPIKE_SIZE)
                y = self.np_random.integers(self.world_rect.top, self.world_rect.bottom - self.SPIKE_SIZE)
                spike_rect = pygame.Rect(x, y, self.SPIKE_SIZE, self.SPIKE_SIZE)
                if not spike_rect.colliderect(player_safe_zone):
                    is_valid_pos = True

            # Create polygon points for drawing
            orientation = self.np_random.integers(0, 4)
            cx, cy = spike_rect.center
            sz = self.SPIKE_SIZE // 2
            if orientation == 0: # Points up
                points = [(cx - sz, cy + sz), (cx + sz, cy + sz), (cx, cy - sz)]
            elif orientation == 1: # Points down
                points = [(cx - sz, cy - sz), (cx + sz, cy - sz), (cx, cy + sz)]
            elif orientation == 2: # Points left
                points = [(cx + sz, cy - sz), (cx + sz, cy + sz), (cx - sz, cy)]
            else: # Points right
                points = [(cx - sz, cy - sz), (cx - sz, cy + sz), (cx + sz, cy)]
            
            self.spikes.append({"rect": spike_rect, "points": points})

    def step(self, action):
        if self.game_over or self.victory:
            # If game is done, just pass time
            movement = 0
        else:
            movement = action[0]
            self._handle_player_movement(movement)
            self.time_remaining_frames -= 1
        
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        reward = self._calculate_reward()
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_rect.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_rect.y += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_rect.x += self.PLAYER_SPEED
        
        # Clamp player to world boundaries
        self.player_rect.clamp_ip(self.world_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _create_collision_particles(self):
        # sfx: player_die.wav
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(self.player_rect.center),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': self.COLOR_PARTICLE
            })

    def _check_termination(self):
        if self.game_over or self.victory:
            return True

        # 1. Collision with spike
        for spike in self.spikes:
            if self.player_rect.colliderect(spike["rect"]):
                self.game_over = True
                self._create_collision_particles()
                return True

        # 2. Victory condition
        if self.time_remaining_frames <= 0:
            self.victory = True
            return True

        # 3. Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True # Treat as a loss
            return True

        return False

    def _calculate_reward(self):
        if self.game_over:
            return -10.0
        if self.victory:
            return 100.0
        return 0.1

    def _get_observation(self):
        self.clock.tick(self.FPS)
        
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw world background
        pygame.draw.rect(self.screen, self.COLOR_WORLD_BG, self.world_rect)

        # Draw spikes
        for spike in self.spikes:
            pygame.draw.polygon(self.screen, self.COLOR_SPIKE, spike["points"])

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

        # Draw player only if not game over
        if not self.game_over:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        # Draw timer
        seconds_left = math.ceil(max(0, self.time_remaining_frames) / self.FPS)
        timer_text = f"Time: {seconds_left}"
        text_surface = self.font_medium.render(timer_text, True, self.COLOR_TIMER)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(text_surface, text_rect)

        # Draw game over/victory message
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_DEFEAT
        elif self.victory:
            msg = "VICTORY!"
            color = self.COLOR_VICTORY
        else:
            return

        text_surface = self.font_large.render(msg, True, color)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": math.ceil(max(0, self.time_remaining_frames) / self.FPS),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Spike Survival")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    print("--- Spike Survival ---")
    print(env.user_guide)
    
    while not terminated:
        # --- Action Mapping for Human ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # The observation is already a rendered frame, so we just need to display it.
        # It's in (H, W, C) format, but pygame surfaces are (W, H), so we need to transpose.
        # The original obs from get_observation is already transposed for gym, we need to reverse it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()