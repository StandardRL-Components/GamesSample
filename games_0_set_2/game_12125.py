import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:07:45.582909
# Source Brief: brief_02125.md
# Brief Index: 2125
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls an ever-extending neon line.
    The goal is to grow the line to a target length of 10 segments by navigating a
    field of obstacles. The line's speed increases with each new segment, creating
    an escalating challenge of speed versus precision.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right. Extends the line.
    - `actions[1]` (Space): No-op.
    - `actions[2]` (Shift): No-op.

    **Observation Space:** A 640x400 RGB image of the game screen.

    **Rewards:**
    - +0.1 per step survived.
    - +1.0 for each successfully added line segment.
    - +20.0 for reaching the target length of 10 segments.
    - -50.0 for colliding with an obstacle.

    **Termination:**
    - The line collides with an obstacle.
    - The line reaches the target length.
    - The episode reaches the maximum step limit (1000).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control an ever-extending neon line to reach a target length. Navigate a field of obstacles as your speed increases with each new segment."
    )
    user_guide = "Use the arrow keys (↑↓←→) to extend the line in a new direction."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_SEGMENTS = 10
    MAX_STEPS = 1000
    SEGMENT_LENGTH = 40
    INITIAL_SPEED = 2.0
    SPEED_INCREMENT = 0.2
    NUM_OBSTACLES = 25
    SAFE_ZONE_RADIUS = 80

    # --- COLORS ---
    COLOR_BG = (15, 15, 25)
    COLOR_LINE = (0, 255, 255)
    COLOR_LINE_GLOW = (0, 100, 100)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_DANGER = (255, 0, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_INACTIVE = (50, 50, 60)
    COLOR_UI_ACTIVE = (0, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.line_points = []
        self.head_pos = pygame.Vector2(0, 0)
        self.target_pos = None
        self.is_extending = False
        self.speed = 0.0
        self.obstacles = []
        self.particles = []
        
        # This call is not strictly necessary but good practice for a fresh env
        # self.reset() # Removed to avoid initializing RNG before super().reset() is called by the runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Line state
        start_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.line_points = [start_pos]
        self.head_pos = start_pos.copy()
        self.target_pos = None
        self.is_extending = False
        self.speed = self.INITIAL_SPEED
        
        # Effects
        self.particles.clear()

        # Procedural Obstacle Generation
        self.obstacles.clear()
        while len(self.obstacles) < self.NUM_OBSTACLES:
            is_circle = self.np_random.random() > 0.5
            if is_circle:
                radius = self.np_random.integers(10, 31)
                pos = pygame.Vector2(
                    self.np_random.integers(radius, self.SCREEN_WIDTH - radius),
                    self.np_random.integers(radius, self.SCREEN_HEIGHT - radius)
                )
                if pos.distance_to(start_pos) > self.SAFE_ZONE_RADIUS + radius:
                    self.obstacles.append({'type': 'circle', 'pos': pos, 'radius': radius})
            else: # Rectangle
                width = self.np_random.integers(20, 61)
                height = self.np_random.integers(20, 61)
                pos = pygame.Vector2(
                    self.np_random.integers(0, self.SCREEN_WIDTH - width),
                    self.np_random.integers(0, self.SCREEN_HEIGHT - height)
                )
                rect = pygame.Rect(pos.x, pos.y, width, height)
                if rect.clipline((start_pos.x - self.SAFE_ZONE_RADIUS, start_pos.y - self.SAFE_ZONE_RADIUS),
                                 (start_pos.x + self.SAFE_ZONE_RADIUS, start_pos.y + self.SAFE_ZONE_RADIUS)):
                    continue # Skip if it overlaps the safe zone rect
                self.obstacles.append({'type': 'rect', 'rect': rect})

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.game_over = False

        # --- 1. Handle Input ---
        movement = action[0]
        if not self.is_extending and movement != 0:
            self._start_new_segment(movement)

        # --- 2. Update Game State ---
        if self.is_extending:
            event_reward, terminated_by_event = self._update_extension()
            reward += event_reward
            if terminated_by_event:
                self.game_over = True
        
        # Survival reward
        if not self.game_over:
            reward += 0.1

        # Update particles
        self._update_particles()

        self.steps += 1
        
        # --- 3. Check Termination Conditions ---
        truncated = self.steps >= self.MAX_STEPS
        terminated = self.game_over or truncated # Truncation is also a form of termination in this context
        
        # --- 4. Return Gym Tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _start_new_segment(self, movement_action):
        """Calculates and sets a new target position based on the action."""
        direction = pygame.Vector2(0, 0)
        if movement_action == 1: direction.y = -1  # Up
        elif movement_action == 2: direction.y = 1   # Down
        elif movement_action == 3: direction.x = -1  # Left
        elif movement_action == 4: direction.x = 1   # Right

        if direction.length() > 0:
            last_point = self.line_points[-1]
            potential_target = last_point + direction * self.SEGMENT_LENGTH

            # Prevent going off-screen
            if (0 < potential_target.x < self.SCREEN_WIDTH and
                0 < potential_target.y < self.SCREEN_HEIGHT):
                self.target_pos = potential_target
                self.is_extending = True

    def _update_extension(self):
        """Moves the head towards the target and checks for events."""
        reward = 0.0
        terminated = False
        
        # If the target and head are at the same position, do nothing to avoid division by zero
        if self.head_pos.distance_squared_to(self.target_pos) == 0:
            return reward, terminated
        
        direction_to_target = (self.target_pos - self.head_pos).normalize()
        self.head_pos += direction_to_target * self.speed
        
        # Check for collision with obstacles
        if self._check_collision(self.head_pos):
            # SFX: Explosion sound
            self._create_particles(self.head_pos, 30, self.COLOR_DANGER)
            reward = -50.0
            terminated = True
            self.is_extending = False # Stop extending on collision
            return reward, terminated

        # Check if target is reached
        if self.head_pos.distance_to(self.target_pos) < self.speed:
            self.head_pos = self.target_pos.copy()
            self.line_points.append(self.head_pos)
            self.is_extending = False
            self.target_pos = None
            
            self.score += 1
            # SFX: Segment complete chime
            reward = 1.0
            
            # Assert speed increase as per brief
            self.speed += self.SPEED_INCREMENT

            # Check for victory
            if len(self.line_points) -1 >= self.TARGET_SEGMENTS:
                # SFX: Victory fanfare
                reward += 20.0
                terminated = True
        
        return reward, terminated
        
    def _check_collision(self, pos):
        """Checks if a point collides with any obstacle."""
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                if obs['rect'].collidepoint(pos.x, pos.y):
                    return True
            elif obs['type'] == 'circle':
                if pos.distance_to(obs['pos']) < obs['radius']:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "line_length": len(self.line_points) -1,
            "speed": self.speed,
        }

    def _render_game(self):
        # Render obstacles
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            elif obs['type'] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, int(obs['pos'].x), int(obs['pos'].y), int(obs['radius']), self.COLOR_OBSTACLE)
                pygame.gfxdraw.filled_circle(self.screen, int(obs['pos'].x), int(obs['pos'].y), int(obs['radius']), self.COLOR_OBSTACLE)

        # Render line with glow
        if len(self.line_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_LINE_GLOW, False, self.line_points, width=7)
            pygame.draw.lines(self.screen, self.COLOR_LINE, False, self.line_points, width=2)

        # Render extending segment
        if self.is_extending:
            segment_points = [self.line_points[-1], self.head_pos]
            pygame.draw.lines(self.screen, self.COLOR_LINE_GLOW, False, segment_points, width=7)
            pygame.draw.lines(self.screen, self.COLOR_LINE, False, segment_points, width=2)
        
        # Render head with glow
        head_pos_int = (int(self.head_pos.x), int(self.head_pos.y))
        pygame.draw.circle(self.screen, self.COLOR_LINE_GLOW, head_pos_int, 8)
        pygame.draw.circle(self.screen, self.COLOR_LINE, head_pos_int, 5)

        # Render particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            radius = int(p['life'] / p['start_life'] * 4)
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], radius)
    
    def _render_ui(self):
        # Render current length
        score_text = self.font_large.render(f"Length: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render target length indicator
        for i in range(self.TARGET_SEGMENTS):
            x = self.SCREEN_WIDTH - 200 + (i * 18)
            y = 15
            rect = pygame.Rect(x, y, 15, 15)
            color = self.COLOR_UI_ACTIVE if i < self.score else self.COLOR_UI_INACTIVE
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
    
    def _create_particles(self, position, count, color):
        for _ in range(count):
            start_life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.Vector2(
                    (self.np_random.random() - 0.5) * 4,
                    (self.np_random.random() - 0.5) * 4
                ),
                'life': start_life,
                'start_life': start_life,
                'color': color
            })
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Key mapping ---
    # Arrows: Extend line
    # R: Reset environment
    # Q: Quit
    
    print("Controls: Arrow keys to extend line. 'R' to reset. 'Q' to quit.")

    display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Line")

    while True:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    env.close()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print(f"Environment reset. Score: {info['score']}")
                
                # Map keys to actions for this step
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4

        # If the key was pressed, take the step. Otherwise, take no-op step to advance time.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Create a display surface and render the observation
        draw_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surf.blit(draw_surf, (0, 0))
        pygame.display.flip()
        
        if reward != 0 and not math.isclose(reward, 0.1): # Print significant rewards
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Steps: {info['steps']}")
            # obs, info = env.reset() # Let the user reset manually with 'R'
        
        env.clock.tick(30) # Run at 30 FPS