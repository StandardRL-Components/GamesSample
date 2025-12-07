import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:11:08.357640
# Source Brief: brief_00317.md
# Brief Index: 317
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a glowing orb to collect points from other orbs. Green orbs are safe, "
        "while red orbs offer a high-risk, high-reward gamble. Reach 100 points to win before time runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to apply thrust and guide your orb around the screen."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game parameters
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    WIN_SCORE = 100
    NUM_GREEN_ORBS = 5
    NUM_RED_ORBS = 2
    
    # Player parameters
    PLAYER_RADIUS = 15
    PLAYER_IMPULSE = 0.6
    PLAYER_FRICTION = 0.985
    PLAYER_TRAIL_LENGTH = 20
    
    # Orb parameters
    ORB_RADIUS = 8
    
    # Particle parameters
    PARTICLE_COUNT = 30
    PARTICLE_LIFESPAN = 25
    PARTICLE_SPEED = 3.0
    
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_GREEN_ORB = (80, 255, 80)
    COLOR_RED_ORB = (255, 80, 80)
    COLOR_WALL = (220, 220, 220)
    COLOR_TEXT = (240, 240, 240)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_trail = deque(maxlen=self.PLAYER_TRAIL_LENGTH)
        
        self.green_orbs = []
        self.red_orbs = []
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # This can be noisy, commenting out for general use

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        # Player state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_trail.clear()
        
        # Orb state
        self.green_orbs = [self._spawn_orb() for _ in range(self.NUM_GREEN_ORBS)]
        self.red_orbs = [self._spawn_orb() for _ in range(self.NUM_RED_ORBS)]

        # Effects state
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action # space_held and shift_held are ignored
        reward = 0.0
        
        # 1. Apply player action
        if movement == 1: self.player_vel[1] -= self.PLAYER_IMPULSE  # Up
        elif movement == 2: self.player_vel[1] += self.PLAYER_IMPULSE # Down
        elif movement == 3: self.player_vel[0] -= self.PLAYER_IMPULSE # Left
        elif movement == 4: self.player_vel[0] += self.PLAYER_IMPULSE # Right
        
        # 2. Update physics
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel
        self.player_trail.append(self.player_pos.copy())
        
        # 3. Handle wall collisions
        wall_hit = False
        if self.player_pos[0] < self.PLAYER_RADIUS:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[:] = 0
            wall_hit = True
        elif self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_RADIUS:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel[:] = 0
            wall_hit = True
        
        if self.player_pos[1] < self.PLAYER_RADIUS:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[:] = 0
            wall_hit = True
        elif self.player_pos[1] > self.SCREEN_HEIGHT - self.PLAYER_RADIUS:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel[:] = 0
            wall_hit = True
            
        if wall_hit:
            reward -= 0.1 # Small penalty for hitting a wall and losing momentum
            # SFX: Wall thump
            
        # 4. Handle orb collisions
        # Green orbs
        for i, orb_pos in enumerate(self.green_orbs):
            if np.linalg.norm(self.player_pos - orb_pos) < self.PLAYER_RADIUS + self.ORB_RADIUS:
                self.score += 5
                reward += 5.0
                self._create_particles(orb_pos, self.COLOR_GREEN_ORB, self.PARTICLE_COUNT)
                self.green_orbs[i] = self._spawn_orb()
                # SFX: Positive chime
        
        # Red orbs
        for i, orb_pos in enumerate(self.red_orbs):
            if np.linalg.norm(self.player_pos - orb_pos) < self.PLAYER_RADIUS + self.ORB_RADIUS:
                self._create_particles(orb_pos, self.COLOR_RED_ORB, self.PARTICLE_COUNT)
                if self.np_random.random() < 0.5:
                    self.score += 20
                    reward += 20.0
                    # SFX: High-value collect
                else:
                    self.score = 0
                    reward -= 10.0
                    # SFX: Negative buzzer, score reset sound
                self.red_orbs[i] = self._spawn_orb()

        # 5. Update particles
        self._update_particles()
        
        # 6. Update game state and check for termination
        self.steps += 1
        
        if self.score >= self.WIN_SCORE:
            self.terminated = True
            reward += 100.0 # Large reward for winning
            # SFX: Victory fanfare
        elif self.steps >= self.MAX_STEPS:
            self.terminated = True
            # SFX: Game over sound
            
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

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
        # Render player trail
        if len(self.player_trail) > 1:
            for i, pos in enumerate(self.player_trail):
                alpha = int(255 * (i / self.PLAYER_TRAIL_LENGTH))
                radius = int(self.PLAYER_RADIUS * (i / self.PLAYER_TRAIL_LENGTH) * 0.5)
                if radius > 1:
                    color = (*self.COLOR_PLAYER, alpha // 3)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)
    
        # Render orbs
        for orb_pos in self.green_orbs:
            self._draw_glowing_circle(orb_pos, self.COLOR_GREEN_ORB, self.ORB_RADIUS)
        for orb_pos in self.red_orbs:
            self._draw_glowing_circle(orb_pos, self.COLOR_RED_ORB, self.ORB_RADIUS)
            
        # Render player
        self._draw_glowing_circle(self.player_pos, self.COLOR_PLAYER, self.PLAYER_RADIUS)
        
        # Render particles
        for p in self.particles:
            life_ratio = p['life'] / self.PARTICLE_LIFESPAN
            radius = int(max(1, p['size'] * life_ratio))
            color = (*p['color'], int(255 * life_ratio))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

        # Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30 # Assuming 30 FPS for display
        timer_text = self.font_large.render(f"Time: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _spawn_orb(self):
        # Spawn away from edges
        padding = 20
        while True:
            pos = self.np_random.uniform(
                low=[padding, padding],
                high=[self.SCREEN_WIDTH - padding, self.SCREEN_HEIGHT - padding],
                size=(2,)
            ).astype(np.float32)
            # Ensure it's not too close to the player's last known position
            if np.linalg.norm(pos - self.player_pos) > self.PLAYER_RADIUS * 4:
                return pos

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(self.PARTICLE_SPEED * 0.5, self.PARTICLE_SPEED)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(self.PARTICLE_LIFESPAN // 2, self.PARTICLE_LIFESPAN),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })
            
    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _draw_glowing_circle(self, pos, color, radius):
        # Draw glow
        glow_radius = int(radius * 1.8)
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), glow_radius, glow_color)
        
        # Draw main circle
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, color)

    def validate_implementation(self):
        print("Running implementation validation...")
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

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # To run with a display, we need to remove the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a persistent key state dictionary for smooth controls
    key_state = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
    }

    # Setup a display window for manual play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Orb Collector")
    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Close the window to quit.")

    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_state:
                    key_state[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in key_state:
                    key_state[event.key] = False

        # --- Action Mapping ---
        movement_action = 0 # No-op
        if key_state[pygame.K_UP]:
            movement_action = 1
        elif key_state[pygame.K_DOWN]:
            movement_action = 2
        elif key_state[pygame.K_LEFT]:
            movement_action = 3
        elif key_state[pygame.K_RIGHT]:
            movement_action = 4

        # The full action must match the MultiDiscrete space
        action = [movement_action, 0, 0] # Space and Shift are not used

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']} in {info['steps']} steps.")
            obs, info = env.reset()
            # In a real scenario you might break the loop, but here we'll just reset and continue
            
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth viewing

    env.close()