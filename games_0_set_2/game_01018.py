
# Generated: 2025-08-27T15:35:59.740021
# Source Brief: brief_01018.md
# Brief Index: 1018

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Avoid the red ghosts and collect the yellow artifacts. Reach the green exit to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Evade spectral pursuers and gather scattered artifacts to escape a haunted house before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Game constants
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.STARTING_LIVES = 3
        self.NUM_ITEMS = 5
        self.NUM_GHOSTS = 3

        # Player properties
        self.PLAYER_SPEED = 4
        self.PLAYER_RADIUS = 8
        self.INVINCIBILITY_DURATION = 2 * self.FPS  # 2 seconds

        # Ghost properties
        self.GHOST_SPEED = 1.5
        self.GHOST_RADIUS = 10
        self.GHOST_TRAIL_LENGTH = 15

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GHOST = (255, 0, 0)
        self.COLOR_ITEM = (255, 220, 0)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_TEXT = (230, 230, 230)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 0
        self.timer = 0
        self.player_pos = [0, 0]
        self.player_invincible_timer = 0
        self.items = []
        self.ghosts = []
        self.walls = []
        self.exit_rect = None
        self.np_random = None
        
        # Initialize state variables
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def _generate_layout(self):
        """Creates the walls and exit for the haunted house."""
        self.walls = []
        wall_thickness = 10
        # Outer boundary
        self.walls.append(pygame.Rect(0, 0, self.WIDTH, wall_thickness))
        self.walls.append(pygame.Rect(0, self.HEIGHT - wall_thickness, self.WIDTH, wall_thickness))
        self.walls.append(pygame.Rect(0, 0, wall_thickness, self.HEIGHT))
        self.walls.append(pygame.Rect(self.WIDTH - wall_thickness, 0, wall_thickness, self.HEIGHT))

        # Internal walls
        self.walls.append(pygame.Rect(100, 100, wall_thickness, 200))
        self.walls.append(pygame.Rect(100, 100, 200, wall_thickness))
        self.walls.append(pygame.Rect(self.WIDTH - 110, 100, wall_thickness, 200))
        self.walls.append(pygame.Rect(self.WIDTH - 300, self.HEIGHT - 110, 200, wall_thickness))
        self.walls.append(pygame.Rect(320 - wall_thickness//2, 0, wall_thickness, 150))
        self.walls.append(pygame.Rect(320 - wall_thickness//2, 250, wall_thickness, 150))

        self.exit_rect = pygame.Rect(self.WIDTH - 40, self.HEIGHT / 2 - 20, 30, 40)

    def _get_valid_spawn_point(self, radius):
        """Finds a random spawn point not inside a wall."""
        while True:
            x = self.np_random.integers(radius, self.WIDTH - radius)
            y = self.np_random.integers(radius, self.HEIGHT - radius)
            spawn_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
            if spawn_rect.collidelist(self.walls) == -1 and not spawn_rect.colliderect(self.exit_rect):
                return [x, y]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = self.STARTING_LIVES
        self.timer = self.MAX_STEPS

        # Layout and Entities
        self._generate_layout()
        self.player_pos = [50, self.HEIGHT / 2]
        self.player_invincible_timer = 0

        # Items
        self.items = []
        for _ in range(self.NUM_ITEMS):
            pos = self._get_valid_spawn_point(10)
            self.items.append(pygame.Rect(pos[0] - 5, pos[1] - 5, 10, 10))

        # Ghosts
        self.ghosts = []
        patrol_patterns = [
            {"type": "horizontal", "y": 50, "x_start": 150, "x_end": self.WIDTH - 150, "dir": 1},
            {"type": "vertical", "x": self.WIDTH / 2, "y_start": 50, "y_end": self.HEIGHT - 50, "dir": 1},
            {"type": "circular", "center": [180, 250], "radius": 60, "angle": 0}
        ]
        for i in range(self.NUM_GHOSTS):
            pattern = patrol_patterns[i % len(patrol_patterns)]
            pos = [0,0]
            if pattern["type"] == "horizontal":
                pos = [pattern["x_start"], pattern["y"]]
            elif pattern["type"] == "vertical":
                pos = [pattern["x"], pattern["y_start"]]
            elif pattern["type"] == "circular":
                angle = self.np_random.uniform(0, 2 * math.pi)
                pattern["angle"] = angle
                pos = [
                    pattern["center"][0] + pattern["radius"] * math.cos(angle),
                    pattern["center"][1] + pattern["radius"] * math.sin(angle)
                ]
            self.ghosts.append({"pos": pos, "pattern": pattern, "trail": []})
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        self.timer -= 1
        reward = -0.01  # Time penalty

        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1

        self._move_player(movement)
        self._update_ghosts()
        reward += self._check_collisions()
        
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.win:
                reward += 100 # Goal-oriented reward for winning
                # sfx: win_fanfare
            elif self.lives <= 0:
                reward -= 100 # Goal-oriented penalty for losing
                # sfx: game_over_sound
            elif self.timer <= 0:
                reward -= 10 # Goal-oriented penalty for timeout
                # sfx: timeout_buzzer
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_player(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -self.PLAYER_SPEED  # Up
        elif movement == 2: dy = self.PLAYER_SPEED   # Down
        elif movement == 3: dx = -self.PLAYER_SPEED  # Left
        elif movement == 4: dx = self.PLAYER_SPEED   # Right

        old_pos = list(self.player_pos)
        player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        player_rect.center = self.player_pos

        self.player_pos[0] += dx
        player_rect.centerx = self.player_pos[0]
        if player_rect.collidelist(self.walls) != -1:
            self.player_pos[0] = old_pos[0]

        self.player_pos[1] += dy
        player_rect.centery = self.player_pos[1]
        if player_rect.collidelist(self.walls) != -1:
            self.player_pos[1] = old_pos[1]
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_ghosts(self):
        for ghost in self.ghosts:
            ghost["trail"].append(list(ghost["pos"]))
            if len(ghost["trail"]) > self.GHOST_TRAIL_LENGTH:
                ghost["trail"].pop(0)

            pattern = ghost["pattern"]
            if pattern["type"] == "horizontal":
                ghost["pos"][0] += self.GHOST_SPEED * pattern["dir"]
                if ghost["pos"][0] >= pattern["x_end"] or ghost["pos"][0] <= pattern["x_start"]:
                    pattern["dir"] *= -1
            elif pattern["type"] == "vertical":
                ghost["pos"][1] += self.GHOST_SPEED * pattern["dir"]
                if ghost["pos"][1] >= pattern["y_end"] or ghost["pos"][1] <= pattern["y_start"]:
                    pattern["dir"] *= -1
            elif pattern["type"] == "circular":
                pattern["angle"] += 0.03
                ghost["pos"][0] = pattern["center"][0] + pattern["radius"] * math.cos(pattern["angle"])
                ghost["pos"][1] = pattern["center"][1] + pattern["radius"] * math.sin(pattern["angle"])

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        player_rect.center = self.player_pos

        # Player-Item
        for i, item_rect in enumerate(self.items):
            if player_rect.colliderect(item_rect):
                self.items.pop(i)
                self.score += 1
                reward += 1 # Event-based reward
                # sfx: item_pickup
                break

        # Player-Ghost
        if self.player_invincible_timer == 0:
            for ghost in self.ghosts:
                dist = math.hypot(self.player_pos[0] - ghost["pos"][0], self.player_pos[1] - ghost["pos"][1])
                if dist < self.PLAYER_RADIUS + self.GHOST_RADIUS:
                    self.lives -= 1
                    self.player_invincible_timer = self.INVINCIBILITY_DURATION
                    self.player_pos = [50, self.HEIGHT / 2]
                    # sfx: player_hit
                    break

        # Player-Exit
        if player_rect.colliderect(self.exit_rect):
            self.win = True
        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.timer <= 0 or self.win or self.steps >= self.MAX_STEPS
    
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
        for wall in self.walls: pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        for ghost in self.ghosts:
            for i, pos in enumerate(ghost["trail"]):
                alpha = int(100 * (i / self.GHOST_TRAIL_LENGTH))
                radius = int(self.GHOST_RADIUS * (i / self.GHOST_TRAIL_LENGTH))
                if radius > 0: pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (*self.COLOR_GHOST, alpha))

        for item_rect in self.items:
            pygame.draw.rect(self.screen, self.COLOR_ITEM, item_rect)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_ITEM), item_rect, 2)

        for ghost in self.ghosts:
            pos = (int(ghost["pos"][0]), int(ghost["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GHOST_RADIUS, (*self.COLOR_GHOST, 150))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GHOST_RADIUS, self.COLOR_GHOST)

        is_visible = self.player_invincible_timer == 0 or (self.player_invincible_timer % 10 < 5)
        if is_visible:
            pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 4, (*self.COLOR_PLAYER, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        time_left = max(0, self.timer / self.FPS)
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (15, 15))

        items_text = self.font_main.render(f"ITEMS: {self.score}/{self.NUM_ITEMS}", True, self.COLOR_TEXT)
        items_rect = items_text.get_rect(centerx=self.WIDTH/2, top=15)
        self.screen.blit(items_text, items_rect)

        heart_size = 20
        for i in range(self.lives):
            pos = (self.WIDTH - 25 - i * (heart_size + 5), 25)
            self._draw_heart(self.screen, pos, heart_size)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg, color = ("YOU ESCAPED!", self.COLOR_EXIT) if self.win else ("GAME OVER", self.COLOR_GHOST)
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _draw_heart(self, surface, pos, size):
        x, y = pos; s = size / 2
        points = [(x, y - s*0.4), (x + s*0.5, y - s), (x + s, y - s*0.7), (x + s, y - s*0.2), (x, y + s), (x - s, y - s*0.2), (x - s, y - s*0.7), (x - s*0.5, y - s)]
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_GHOST)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_GHOST)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_left_steps": self.timer,
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Main block for testing
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print("Initial state:", info)

    terminated = False
    total_reward = 0
    for i in range(2000): # Run for more than max steps to test termination
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 300 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
        if terminated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}, Final Total Reward: {total_reward:.2f}")
            # Example of saving a frame on termination
            # from PIL import Image
            # img = Image.fromarray(obs)
            # img.save(f"terminated_frame_{i+1}.png")
            break
    env.close()