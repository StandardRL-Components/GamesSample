
# Generated: 2025-08-27T19:18:47.114768
# Source Brief: brief_02114.md
# Brief Index: 2114

        
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
    """
    A Gymnasium environment for a fast-paced arcade game.
    The player must catch all scurrying critters on a grid before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move. Catch all the critters before time runs out!"
    )

    # User-facing game description
    game_description = (
        "Fast-paced arcade action. Scurry around the grid to catch all the critters before the timer hits zero!"
    )

    # Frames advance only on action, as the game is turn-based.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_CRITTERS = 25
        self.TIME_LIMIT_SECONDS = 60.0
        self.TIME_PER_STEP = 0.25
        self.MAX_STEPS = int(self.TIME_LIMIT_SECONDS / self.TIME_PER_STEP)

        # Visual constants
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.OFFSET_Y = self.HEIGHT - self.GRID_HEIGHT - 10

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_CRITTER = (255, 50, 50)
        self.COLOR_CRITTER_OUTLINE = (255, 150, 150)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_SPARKLE = [(255, 255, 255), (255, 255, 150), (255, 200, 0)]

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
        self.font_ui = pygame.font.Font(None, 32)
        self.font_timer = pygame.font.Font(None, 40)

        # Initialize state variables
        self.player_pos = None
        self.critters = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_left = 0.0
        self.game_over = False
        self.rng = None
        
        # Validate implementation after setup
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.particles = []

        # Place player in the center
        self.player_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])

        # Generate critters
        self.critters = []
        occupied_pos = {tuple(self.player_pos)}
        for _ in range(self.NUM_CRITTERS):
            while True:
                pos = tuple(self.rng.integers(0, self.GRID_SIZE, size=2))
                if pos not in occupied_pos:
                    occupied_pos.add(pos)
                    break
            
            critter = {
                "pos": np.array(pos),
                "pattern": self.rng.integers(0, 5, size=100),
                "pattern_idx": 0,
                "turns_until_change": self.rng.integers(1, 4),
                "current_direction": 0,
            }
            self.critters.append(critter)

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game by one step.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0

        # 1. Calculate pre-move distance for reward shaping
        dist_before = self._get_dist_to_nearest_critter()

        # 2. Update player position
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        self.player_pos = np.clip(self.player_pos, 0, self.GRID_SIZE - 1)

        # 3. Update critter positions
        for critter in self.critters:
            critter["turns_until_change"] -= 1
            if critter["turns_until_change"] <= 0:
                critter["current_direction"] = critter["pattern"][critter["pattern_idx"]]
                critter["pattern_idx"] = (critter["pattern_idx"] + 1) % len(critter["pattern"])
                critter["turns_until_change"] = self.rng.integers(1, 4)

            c_move = critter["current_direction"]
            if c_move == 1: critter["pos"][1] -= 1
            elif c_move == 2: critter["pos"][1] += 1
            elif c_move == 3: critter["pos"][0] -= 1
            elif c_move == 4: critter["pos"][0] += 1
            critter["pos"] = np.clip(critter["pos"], 0, self.GRID_SIZE - 1)

        # 4. Check for catches and apply event-based reward
        remaining_critters = []
        for critter in self.critters:
            if np.array_equal(self.player_pos, critter["pos"]):
                reward += 10.0  # +10 for catching a critter
                self._create_sparkles(critter["pos"])
                # sfx: catch_critter.wav
            else:
                remaining_critters.append(critter)
        self.critters = remaining_critters

        # 5. Calculate movement-based reward
        dist_after = self._get_dist_to_nearest_critter()
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 1.0  # +1 for moving closer
            elif dist_after > dist_before:
                reward -= 0.1  # -0.1 for moving further

        self.score += reward

        # 6. Update game state
        self.steps += 1
        self.time_left -= self.TIME_PER_STEP

        # 7. Update particles
        self._update_particles()

        # 8. Check for termination
        terminated = False
        if not self.critters:
            terminated = True
            reward += 50.0  # +50 win bonus
            # sfx: win_game.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 50.0  # -50 lose penalty
            # sfx: lose_game.wav
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        """
        Renders the current game state to a numpy array.
        """
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_critters()
        self._render_player()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "critters_left": len(self.critters),
        }

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        px = self.OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _get_dist_to_nearest_critter(self):
        """Calculates Manhattan distance to the nearest critter."""
        if not self.critters:
            return None
        
        player_pos_rep = np.tile(self.player_pos, (len(self.critters), 1))
        critter_positions = np.array([c["pos"] for c in self.critters])
        distances = np.sum(np.abs(player_pos_rep - critter_positions), axis=1)
        return np.min(distances)

    def _create_sparkles(self, grid_pos):
        """Creates a burst of particles for a catch effect."""
        pixel_pos = self._grid_to_pixel(grid_pos)
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(10, 20)
            self.particles.append({
                "pos": list(pixel_pos),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": random.choice(self.COLOR_SPARKLE),
            })

    def _update_particles(self):
        """Updates position and life of all active particles."""
        active_particles = []
        for p in self.particles:
            p["life"] -= 1
            if p["life"] > 0:
                p["pos"][0] += p["vel"][0]
                p["pos"][1] += p["vel"][1]
                p["vel"][0] *= 0.95 # friction
                p["vel"][1] *= 0.95
                active_particles.append(p)
        self.particles = active_particles

    def _render_grid(self):
        """Renders the game grid."""
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y)
            end_pos = (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.OFFSET_X, self.OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.OFFSET_X + self.GRID_WIDTH, self.OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_player(self):
        """Renders the player character."""
        px, py = self._grid_to_pixel(self.player_pos)
        radius = self.CELL_SIZE // 2 - 4
        points = [
            (px - radius, py - radius),
            (px + radius, py - radius),
            (px + radius, py + radius),
            (px - radius, py + radius),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
    
    def _render_critters(self):
        """Renders all critters."""
        radius = self.CELL_SIZE // 2 - 6
        for critter in self.critters:
            px, py = self._grid_to_pixel(critter["pos"])
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_CRITTER_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_CRITTER)

    def _render_particles(self):
        """Renders all active particles."""
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(life_ratio * 5)
            if radius > 0:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                color = tuple(int(c * life_ratio) for c in p["color"])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        """Renders the UI elements (score, time, etc.)."""
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Critters left
        critter_text = self.font_ui.render(f"Critters: {len(self.critters)}/{self.NUM_CRITTERS}", True, self.COLOR_UI_TEXT)
        critter_rect = critter_text.get_rect(center=(self.WIDTH // 2, 10 + score_text.get_height() // 2))
        self.screen.blit(critter_text, critter_rect)

        # Timer
        time_str = f"{max(0, self.time_left):.2f}"
        timer_color = (255, 80, 80) if self.time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_timer.render(time_str, True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

    def close(self):
        """Closes the Pygame window."""
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    env.reset()
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if not terminated:
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

        # In a real-time game, you'd step every frame.
        # Since this is turn-based, we only step on key press.
        # For this demo, we'll step continuously if a key is held.
        if action[0] != 0 or not terminated: # Allow no-op steps to pass time
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the main display
        display_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Critter Catcher")
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(render_surface, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a moment then reset
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

        env.clock.tick(15) # Control the speed of the manual play

    env.close()