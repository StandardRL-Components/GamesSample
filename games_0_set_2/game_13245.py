import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls an oscillating particle.
    The goal is to navigate a field of moving obstacles to reach a target zone.
    The agent controls the amplitude and frequency of the particle's sine wave motion.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Fixes for tests ---
    game_description = (
        "Control an oscillating particle, navigating a field of moving obstacles to reach the target zone."
    )
    user_guide = (
        "Controls: Use ↑/↓ arrows to change the particle's oscillation frequency and ←/→ to change its amplitude."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60-second time limit

        # Colors
        self.COLOR_BG = (26, 26, 46) # Dark blue/purple
        self.COLOR_OBSTACLE = (255, 70, 100)
        self.COLOR_TARGET = (80, 255, 120)
        self.COLOR_BOOST = (0, 150, 255)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_TRAIL = (255, 255, 255)

        # Player Particle
        self.PARTICLE_RADIUS = 10
        self.INITIAL_SPEED = 2.0
        self.BOOSTED_SPEED = 4.0
        self.MIN_AMPLITUDE, self.MAX_AMPLITUDE = 10, (self.HEIGHT // 2) - self.PARTICLE_RADIUS - 10
        self.MIN_FREQUENCY, self.MAX_FREQUENCY = 0.5, 5.0
        self.AMP_STEP = 2.0
        self.FREQ_STEP = 0.1

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 16)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_elapsed = None
        
        self.particle_pos = None
        self.particle_y_center = None
        self.particle_speed = None
        self.amplitude = None
        self.frequency = None
        
        self.boost_collected = None
        self.boost_tile_rect = None
        
        self.target_rect = None
        self.obstacles = None
        self.particle_trail = None
        self.static_stars = None

        # --- Final setup ---
        if render_mode == "human":
            pygame.display.set_caption("Oscillator")
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.render_mode = render_mode
        # Note: Environment is not reset in __init__ per Gymnasium standard practice.
        # The user must call reset() before using the environment.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_elapsed = 0.0

        # Player particle
        self.particle_y_center = self.HEIGHT / 2
        self.particle_pos = np.array([70.0, self.particle_y_center])
        self.particle_speed = self.INITIAL_SPEED
        self.amplitude = 50.0
        self.frequency = 1.5
        self.particle_trail = deque(maxlen=100)

        # Boost tile
        self.boost_collected = False
        self.boost_tile_rect = pygame.Rect(self.WIDTH * 0.4, self.HEIGHT * 0.7, 40, 40)

        # Target
        self.target_rect = pygame.Rect(self.WIDTH - 40, 0, 40, self.HEIGHT)

        # Obstacles
        self.obstacles = self._spawn_obstacles(num_obstacles=10)
        
        # Background stars (FIX: use self.np_random for reproducibility)
        self.static_stars = [
            (self.np_random.integers(0, self.WIDTH + 1), self.np_random.integers(0, self.HEIGHT + 1), self.np_random.integers(1, 3))
            for _ in range(100)
        ]

        return self._get_observation(), self._get_info()

    def _spawn_obstacles(self, num_obstacles):
        obstacles = []
        # FIX: Pushed spawn area to the right to create a safe zone at the start,
        # preventing termination during the stability test.
        spawn_area_x = (self.WIDTH * 0.35, self.WIDTH * 0.85)
        for _ in range(num_obstacles):
            size = self.np_random.integers(15, 35)
            x = self.np_random.uniform(spawn_area_x[0], spawn_area_x[1])
            y = self.np_random.uniform(0, self.HEIGHT - size)
            rect = pygame.Rect(int(x), int(y), size, size)
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            
            obstacles.append({'rect': rect, 'vel': velocity, 'size': size})
        return obstacles

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_action(action)
        self._update_game_state()

        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        self.game_over = terminated

        # FIX: Handle truncation separately from termination, per Gymnasium API.
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]

        if movement == 1:  # Increase Frequency
            self.frequency = min(self.MAX_FREQUENCY, self.frequency + self.FREQ_STEP)
        elif movement == 2:  # Decrease Frequency
            self.frequency = max(self.MIN_FREQUENCY, self.frequency - self.FREQ_STEP)
        elif movement == 3:  # Increase Amplitude
            self.amplitude = min(self.MAX_AMPLITUDE, self.amplitude + self.AMP_STEP)
        elif movement == 4:  # Decrease Amplitude
            self.amplitude = max(self.MIN_AMPLITUDE, self.amplitude - self.AMP_STEP)
        # movement == 0 is no-op

    def _update_game_state(self):
        self.steps += 1
        self.time_elapsed += 1 / self.FPS

        # Update particle
        self.particle_pos[0] += self.particle_speed
        self.particle_pos[1] = self.particle_y_center + self.amplitude * math.sin(self.frequency * 2 * math.pi * self.time_elapsed)
        self.particle_trail.append(tuple(self.particle_pos.astype(int)))
        
        # Update obstacles
        for obs in self.obstacles:
            obs['rect'].move_ip(obs['vel'])
            if obs['rect'].left < 0 or obs['rect'].right > self.WIDTH:
                obs['vel'][0] *= -1
            if obs['rect'].top < 0 or obs['rect'].bottom > self.HEIGHT:
                obs['vel'][1] *= -1

    def _calculate_reward_and_termination(self):
        reward = 0.0
        terminated = False

        particle_rect = pygame.Rect(
            self.particle_pos[0] - self.PARTICLE_RADIUS,
            self.particle_pos[1] - self.PARTICLE_RADIUS,
            self.PARTICLE_RADIUS * 2,
            self.PARTICLE_RADIUS * 2
        )

        # Win condition
        if particle_rect.colliderect(self.target_rect):
            return 100.0, True

        # Obstacle collision
        for obs in self.obstacles:
            if particle_rect.colliderect(obs['rect']):
                return -100.0, True
        
        # Boost tile interaction
        if not self.boost_collected and particle_rect.colliderect(self.boost_tile_rect):
            self.boost_collected = True
            self.particle_speed = self.BOOSTED_SPEED
            reward += 5.0

        # Survival reward
        reward += 0.1
        
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw static stars
        for x, y, size in self.static_stars:
            pygame.draw.rect(self.screen, (60, 60, 80), (x, y, size, size))
            
        # Draw target zone (semi-transparent)
        target_surface = pygame.Surface((self.target_rect.width, self.target_rect.height), pygame.SRCALPHA)
        target_surface.fill((*self.COLOR_TARGET, 50))
        self.screen.blit(target_surface, self.target_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_TARGET, self.target_rect, 2)

        # Draw boost tile
        if not self.boost_collected:
            boost_surface = pygame.Surface((self.boost_tile_rect.width, self.boost_tile_rect.height), pygame.SRCALPHA)
            boost_surface.fill((*self.COLOR_BOOST, 150))
            self.screen.blit(boost_surface, self.boost_tile_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_BOOST, self.boost_tile_rect, 2)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'], border_radius=3)

        # Draw particle trail
        if len(self.particle_trail) > 1:
            # pygame.draw.lines with alpha is not directly supported, this is a simplified version
            # that works by drawing on a separate surface for each segment (can be slow).
            # The original code's alpha value in the color tuple was ignored by pygame.draw.line.
            for i in range(len(self.particle_trail) - 1):
                alpha = int(200 * (i / len(self.particle_trail)))
                try:
                    # This method of drawing transparent lines is more correct but can be slow.
                    line_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(line_surf, (*self.COLOR_TRAIL, alpha), self.particle_trail[i], self.particle_trail[i+1], 2)
                    self.screen.blit(line_surf, (0,0))
                except (TypeError, ValueError):
                    # Fallback for invalid trail points during edge cases
                    pass

        # Draw particle with glow
        px, py = int(self.particle_pos[0]), int(self.particle_pos[1])
        glow_radius = int(self.PARTICLE_RADIUS * 1.8)
        glow_color = (*self.COLOR_PLAYER, 50)
        
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, glow_color)
        self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius))

        pygame.gfxdraw.aacircle(self.screen, px, py, self.PARTICLE_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PARTICLE_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Player stats
        amp_text = f"Amp: {self.amplitude:.1f}"
        freq_text = f"Freq: {self.frequency:.1f} Hz"
        speed_text = f"Speed: {self.particle_speed:.1f}x"
        
        amp_surf = self.font_info.render(amp_text, True, self.COLOR_UI_TEXT)
        freq_surf = self.font_info.render(freq_text, True, self.COLOR_UI_TEXT)
        speed_surf = self.font_info.render(speed_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(amp_surf, (10, self.HEIGHT - 50))
        self.screen.blit(freq_surf, (10, self.HEIGHT - 35))
        self.screen.blit(speed_surf, (10, self.HEIGHT - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "boost_collected": self.boost_collected
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Example usage: Run the game with a human player or random agent
    # Use arrow keys to control: UP/DOWN for frequency, LEFT/RIGHT for amplitude
    
    env = GameEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up(freq+), 2=down(freq-), 3=right(amp+), 4=left(amp-)
    
    print("--- Controls ---")
    print("UP/DOWN Arrows: Change Frequency")
    print("LEFT/RIGHT Arrows: Change Amplitude (Right=Increase, Left=Decrease)")
    print("----------------")
    
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        movement_action = 0
        if keys[pygame.K_UP]:
            movement_action = 1 # Increase Freq
        elif keys[pygame.K_DOWN]:
            movement_action = 2 # Decrease Freq
        elif keys[pygame.K_RIGHT]:
            movement_action = 3 # Increase Amp
        elif keys[pygame.K_LEFT]:
            movement_action = 4 # Decrease Amp
            
        action[0] = movement_action
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            done = True

    env.close()