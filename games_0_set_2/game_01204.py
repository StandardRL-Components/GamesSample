
# Generated: 2025-08-27T16:23:06.283604
# Source Brief: brief_01204.md
# Brief Index: 1204

        
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


# Helper class for particle effects
class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        self.vx = self.np_random.uniform(-2, 2)
        self.vy = self.np_random.uniform(-4, -1)
        self.life = 20  # Lifetime in frames

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # Gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), max(0, int(self.life / 5)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your squirrel one tile at a time. "
        "Avoid the guards' red vision cones."
    )

    game_description = (
        "A top-down stealth game. Guide the squirrel to the acorn without being spotted by the patrolling guards. "
        "Each move is a turn. Plan your path carefully!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.TILE_SIZE = self.WIDTH // self.GRID_WIDTH
        self.MAX_TURNS = 100
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (40, 45, 50)
        self.COLOR_GRID = (60, 65, 70)
        self.COLOR_WALL = (80, 90, 100)
        self.COLOR_SQUIRREL = (180, 120, 90)
        self.COLOR_SQUIRREL_OUTLINE = (255, 255, 255)
        self.COLOR_GUARD = (220, 50, 50)
        self.COLOR_GUARD_OUTLINE = (255, 150, 150)
        self.COLOR_VISION_CONE = (200, 60, 60, 100)
        self.COLOR_ACORN = (255, 220, 50)
        self.COLOR_GOAL_TILE = (60, 100, 60)
        self.COLOR_TEXT = (240, 240, 240)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # --- Game State (persistent across episodes) ---
        self.total_episodes = 0
        self.base_guard_speed = 0.5
        
        # --- Game State (reset each episode) ---
        self.squirrel_pos = None
        self.acorn_pos = None
        self.walls = []
        self.guards = []
        self.particles = []
        
        self.steps = 0
        self.turns_left = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_episodes += 1
        
        # --- Reset game state variables ---
        self.steps = 0
        self.score = 0
        self.turns_left = self.MAX_TURNS
        self.game_over = False
        self.game_over_message = ""
        self.particles = []

        # --- Difficulty Scaling ---
        speed_increase = (self.total_episodes // 500) * 0.05
        self.guard_speed = self.base_guard_speed + speed_increase

        # --- Level Generation ---
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        self.walls = []
        # Create borders
        for x in range(self.GRID_WIDTH):
            self.walls.append((x, 0))
            self.walls.append((x, self.GRID_HEIGHT - 1))
        for y in range(1, self.GRID_HEIGHT - 1):
            self.walls.append((0, y))
            self.walls.append((self.GRID_WIDTH - 1, y))
            
        # Add some internal walls
        for y in range(5, 15):
            self.walls.append((10, y))
            self.walls.append((22, y))
        for x in range(10, 23):
            self.walls.append((x, 10))

        # Generate valid positions for items
        valid_positions = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x,y) not in self.walls:
                    valid_positions.append((x,y))
        
        # Place squirrel and acorn
        start_end_indices = self.np_random.choice(len(valid_positions), 2, replace=False)
        self.squirrel_pos = valid_positions[start_end_indices[0]]
        self.acorn_pos = valid_positions[start_end_indices[1]]

        # --- Initialize Guards ---
        self.guards = []
        guard_paths = [
            [(2, 2), (2, self.GRID_HEIGHT - 2)],
            [(self.GRID_WIDTH - 2, 2), (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)],
            [(11, 2), (21, 2)],
            [(11, self.GRID_HEIGHT - 2), (21, self.GRID_HEIGHT - 2)],
        ]
        
        for path in guard_paths:
            start_pos = path[0]
            self.guards.append({
                "pos": np.array(start_pos, dtype=float),
                "path": path,
                "target_idx": 1,
                "direction": np.array([0, 1.0]),
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = -0.1  # Time penalty
        self.steps += 1
        self.turns_left -= 1
        
        # --- Player Movement ---
        prev_pos = self.squirrel_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            next_pos = (self.squirrel_pos[0] + dx, self.squirrel_pos[1] + dy)
            if next_pos not in self.walls:
                self.squirrel_pos = next_pos
        
        # --- Guard Movement & Logic ---
        self._update_guards()

        # --- Check Game State and Calculate Rewards ---
        terminated = False
        
        # Win Condition
        if self.squirrel_pos == self.acorn_pos:
            reward += 100
            self.game_over_message = "ACORN GET!"
            terminated = True
            # Sound: Win jingle
            for _ in range(50): self.particles.append(Particle(self.squirrel_pos[0]*self.TILE_SIZE + self.TILE_SIZE/2, self.squirrel_pos[1]*self.TILE_SIZE + self.TILE_SIZE/2, self.COLOR_ACORN, self.np_random))

        # Loss Conditions
        if not terminated:
            caught, spotted = self._check_detection()
            if caught or spotted:
                reward = -100
                self.game_over_message = "SPOTTED!"
                terminated = True
                # Sound: Detection alert
                for _ in range(30): self.particles.append(Particle(self.squirrel_pos[0]*self.TILE_SIZE + self.TILE_SIZE/2, self.squirrel_pos[1]*self.TILE_SIZE + self.TILE_SIZE/2, self.COLOR_GUARD, self.np_random))
            elif self.squirrel_pos != prev_pos: # Only reward for moving to a safe tile
                reward += 1.0

        # Timeout or Max Steps
        if self.turns_left <= 0:
            self.game_over_message = "TIME OUT!"
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        if terminated:
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_guards(self):
        for guard in self.guards:
            target_pos = np.array(guard["path"][guard["target_idx"]], dtype=float)
            current_pos = guard["pos"]
            
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            
            if distance < self.guard_speed:
                guard["pos"] = target_pos
                guard["target_idx"] = 1 - guard["target_idx"] # Flip between 0 and 1
            else:
                normalized_direction = direction / distance
                guard["pos"] += normalized_direction * self.guard_speed
                guard["direction"] = normalized_direction

    def _check_detection(self):
        squirrel_center = (
            (self.squirrel_pos[0] + 0.5) * self.TILE_SIZE,
            (self.squirrel_pos[1] + 0.5) * self.TILE_SIZE
        )

        for guard in self.guards:
            # Direct collision check
            guard_tile = (int(guard["pos"][0]), int(guard["pos"][1]))
            if self.squirrel_pos == guard_tile:
                return True, False # Caught

            # Vision cone check
            p1 = guard["pos"] * self.TILE_SIZE + self.TILE_SIZE / 2
            
            direction = guard["direction"]
            perp_dir = np.array([-direction[1], direction[0]])
            
            cone_length = 6 * self.TILE_SIZE
            cone_width = 3 * self.TILE_SIZE
            
            p2 = p1 + direction * cone_length + perp_dir * cone_width
            p3 = p1 + direction * cone_length - perp_dir * cone_width
            
            if self._is_point_in_triangle(squirrel_center, p1, p2, p3):
                return False, True # Spotted
                
        return False, False
        
    def _is_point_in_triangle(self, pt, v1, v2, v3):
        # Barycentric coordinate system check
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw goal tile
        pygame.draw.rect(self.screen, self.COLOR_GOAL_TILE, (self.acorn_pos[0] * self.TILE_SIZE, self.acorn_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (wall[0] * self.TILE_SIZE, wall[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw acorn
        acorn_center = (int(self.acorn_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2), int(self.acorn_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2))
        pygame.draw.circle(self.screen, self.COLOR_ACORN, acorn_center, int(self.TILE_SIZE * 0.4))
        
        # Draw guards and vision cones
        for guard in self.guards:
            # Vision cone
            p1 = guard["pos"] * self.TILE_SIZE + self.TILE_SIZE / 2
            
            direction = guard["direction"]
            # Ensure direction is not zero vector
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([0, 1.0]) # Default direction

            perp_dir = np.array([-direction[1], direction[0]])
            
            cone_length = 6 * self.TILE_SIZE
            cone_width = 3 * self.TILE_SIZE
            
            p2 = p1 + direction * cone_length + perp_dir * cone_width
            p3 = p1 + direction * cone_length - perp_dir * cone_width
            
            points = [tuple(p1), tuple(p2), tuple(p3)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_VISION_CONE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_VISION_CONE)
            
            # Guard body
            guard_center = (int(p1[0]), int(p1[1]))
            radius = int(self.TILE_SIZE * 0.4)
            pygame.draw.circle(self.screen, self.COLOR_GUARD, guard_center, radius)
            pygame.draw.circle(self.screen, self.COLOR_GUARD_OUTLINE, guard_center, radius, 1)

        # Draw squirrel
        squirrel_center = (int(self.squirrel_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2), int(self.squirrel_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2))
        radius = int(self.TILE_SIZE * 0.4)
        pygame.draw.circle(self.screen, self.COLOR_SQUIRREL, squirrel_center, radius)
        pygame.draw.circle(self.screen, self.COLOR_SQUIRREL_OUTLINE, squirrel_center, radius, 2)
        
        # Draw particles
        for p in self.particles:
            p.update()
            p.draw(self.screen)
        self.particles = [p for p in self.particles if p.life > 0]

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 5))
        
        # Turns left
        turns_text = self.font_small.render(f"TURNS: {self.turns_left}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (10, 5))

        # Game Over Message
        if self.game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            over_text = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns_left": self.turns_left,
            "squirrel_pos": self.squirrel_pos,
            "acorn_pos": self.acorn_pos
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        import sys
        pygame.display.set_caption("Stealth Squirrel")
        render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        obs, info = env.reset()
        done = False
        
        while True:
            action = [0, 0, 0] # Default: no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        done = False
                        
            if not env.game_over:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                
                # We only step if a movement key is pressed, simulating turns
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            
            # Render the environment's surface to the display
            draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(draw_surface, (0, 0))
            pygame.display.flip()
            
            # Since it's turn-based, we wait for input, but cap the loop speed
            pygame.time.wait(30)
            
    except pygame.error as e:
        print(f"\nPygame display error: {e}")
        print("This error is expected in a headless environment. The environment itself is functional.")
        print("The 'if __name__ == '__main__':' block is for human testing with a display.")
    
    env.close()