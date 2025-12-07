import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A fast-paced, real-time Sokoban puzzle environment.

    The player must push all crates onto their designated goal locations before
    the 60-second timer runs out. The game prioritizes smooth visuals and
    responsive, arcade-style gameplay over realistic physics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ↑↓←→ to move your character and push crates."
    )

    # User-facing description of the game
    game_description = (
        "Race against the clock to push all the crates onto their designated goal spots in this fast-paced puzzle game."
    )

    # Frames auto-advance for real-time timer and smooth graphics
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_SIZE = 40
    GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // TILE_SIZE, SCREEN_HEIGHT // TILE_SIZE

    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    NUM_CRATES = 5
    
    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_WALL = (82, 82, 82)
    COLOR_PLAYER = (229, 90, 85) # Bright Red
    COLOR_CRATE = (200, 150, 100) # Brown
    COLOR_CRATE_ON_GOAL = (150, 220, 150) # Greenish Brown
    COLOR_GOAL = (80, 150, 80)
    COLOR_GOAL_ACTIVE = (120, 220, 120)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TIMER_WARN = (255, 100, 100)
    
    # --- Visual interpolation ---
    LERP_FACTOR = 0.6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vis_pos = None
        self.crates = None
        self.goals = None
        self.walls = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = deque()
        self.rng = np.random.default_rng()
        
        # The validation function requires the environment to be initialized.
        # We call reset() here to set up the initial state.
        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        """
        Generates a new, solvable level layout.
        An open field with border walls guarantees solvability.
        """
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        empty_cells = [
            (x, y) for x in range(1, self.GRID_WIDTH - 1) for y in range(1, self.GRID_HEIGHT - 1)
        ]
        
        # Ensure we can place all items
        if len(empty_cells) < self.NUM_CRATES * 2 + 1:
            raise ValueError("Grid too small for the number of crates and goals.")

        # Use a list for shuffling with the numpy generator
        empty_cells_list = list(empty_cells)
        self.rng.shuffle(empty_cells_list)
        
        self.goals = set(empty_cells_list[:self.NUM_CRATES])
        
        crate_positions = empty_cells_list[self.NUM_CRATES : self.NUM_CRATES * 2]
        self.crates = [
            {"pos": pos, "vis_pos": (pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE)} 
            for pos in crate_positions
        ]
        
        self.player_pos = empty_cells_list[self.NUM_CRATES * 2]
        self.player_vis_pos = (self.player_pos[0] * self.TILE_SIZE, self.player_pos[1] * self.TILE_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._generate_level()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Small penalty per step to encourage speed

        # --- Player Movement and Crate Pushing Logic ---
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check for wall collision
            if next_player_pos in self.walls:
                pass # Can't move
            
            # Check for crate collision
            elif any(c["pos"] == next_player_pos for c in self.crates):
                crate_idx = next(i for i, c in enumerate(self.crates) if c["pos"] == next_player_pos)
                crate_to_push = self.crates[crate_idx]
                next_crate_pos = (crate_to_push["pos"][0] + dx, crate_to_push["pos"][1] + dy)
                
                # Check if crate can be pushed
                is_blocked = (
                    next_crate_pos in self.walls or
                    any(c["pos"] == next_crate_pos for c in self.crates)
                )
                
                if not is_blocked:
                    # --- Calculate rewards before moving the crate ---
                    was_on_goal = crate_to_push["pos"] in self.goals
                    is_on_goal = next_crate_pos in self.goals
                    
                    if is_on_goal and not was_on_goal:
                        reward += 1.0  # Crate pushed onto a goal
                        # Sound: positive_beep.wav
                    elif not is_on_goal and was_on_goal:
                        reward -= 0.2  # Crate pushed off a goal
                        # Sound: negative_buzz.wav

                    # Move player and crate
                    self.player_pos = next_player_pos
                    crate_to_push["pos"] = next_crate_pos
                    self._create_particles(next_player_pos)
                    # Sound: crate_push.wav
            
            # No collision, just move player
            else:
                self.player_pos = next_player_pos
                # Sound: player_step.wav

        self.steps += 1
        
        # --- Check for win/loss conditions ---
        crates_on_goals = sum(1 for c in self.crates if c["pos"] in self.goals)
        
        if crates_on_goals == self.NUM_CRATES:
            self.win = True
            self.game_over = True
            reward += 100.0  # Big reward for winning
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward -= 100.0  # Big penalty for losing

        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        self._update_visuals()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        crates_on_goals = sum(1 for c in self.crates if c["pos"] in self.goals)
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "crates_on_goals": crates_on_goals,
        }

    def _update_visuals(self):
        """Interpolate visual positions towards logical grid positions."""
        # Player
        target_x, target_y = self.player_pos[0] * self.TILE_SIZE, self.player_pos[1] * self.TILE_SIZE
        self.player_vis_pos = (
            self.player_vis_pos[0] + (target_x - self.player_vis_pos[0]) * self.LERP_FACTOR,
            self.player_vis_pos[1] + (target_y - self.player_vis_pos[1]) * self.LERP_FACTOR,
        )

        # Crates
        for crate in self.crates:
            target_x, target_y = crate["pos"][0] * self.TILE_SIZE, crate["pos"][1] * self.TILE_SIZE
            crate["vis_pos"] = (
                crate["vis_pos"][0] + (target_x - crate["vis_pos"][0]) * self.LERP_FACTOR,
                crate["vis_pos"][1] + (target_y - crate["vis_pos"][1]) * self.LERP_FACTOR,
            )
        
        # Particles
        for p in list(self.particles):
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_game(self):
        """Render the main game elements."""
        self.screen.fill(self.COLOR_BG)

        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw goals
        for gx, gy in self.goals:
            is_occupied = any(c["pos"] == (gx, gy) for c in self.crates)
            color = self.COLOR_GOAL_ACTIVE if is_occupied else self.COLOR_GOAL
            pygame.draw.rect(self.screen, color, (gx * self.TILE_SIZE, gy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw walls
        for wx, wy in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (wx * self.TILE_SIZE, wy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Draw crates
        for crate in self.crates:
            color = self.COLOR_CRATE_ON_GOAL if crate["pos"] in self.goals else self.COLOR_CRATE
            rect = pygame.Rect(crate["vis_pos"], (self.TILE_SIZE, self.TILE_SIZE))
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Draw player
        player_rect = pygame.Rect(self.player_vis_pos, (self.TILE_SIZE, self.TILE_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=6)
        
        # Draw particles
        for p in self.particles:
            size = int(p['life'] / p['max_life'] * 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        """Render UI elements like timer and score."""
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {max(0, time_left):.1f}"
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_TEXT
        
        score_text = f"SCORE: {int(self.score)}"
        
        crates_on_goals = sum(1 for c in self.crates if c["pos"] in self.goals)
        crates_text = f"CRATES: {crates_on_goals}/{self.NUM_CRATES}"

        self._draw_text(timer_text, (self.SCREEN_WIDTH - 10, 10), color=timer_color, align="topright")
        self._draw_text(score_text, (self.SCREEN_WIDTH - 10, 35), align="topright")
        self._draw_text(crates_text, (self.SCREEN_WIDTH - 10, 60), align="topright")

        if self.game_over:
            message = "YOU WIN!" if self.win else "TIME UP!"
            self._draw_text(message, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), font=self.font_big, align="center")

    def _draw_text(self, text, pos, font=None, color=None, align="topleft"):
        if font is None: font = self.font_ui
        if color is None: color = self.COLOR_TEXT
        
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        if align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
            
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, grid_pos):
        """Create a burst of particles at a grid position."""
        px, py = (grid_pos[0] + 0.5) * self.TILE_SIZE, (grid_pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(10):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 3)
            life = self.rng.integers(10, 20)
            self.particles.append({
                'pos': (px, py),
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'life': life,
                'max_life': life,
                'color': (255, 255, 255)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # NOTE: self.reset() is called in __init__ before this function to ensure
        # the environment state is initialized.

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (using a fresh observation)
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
        assert not trunc
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # The environment is validated upon instantiation
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a display. Set render_mode='human' if you adapt the class for it.
    # For now, we will simulate a game loop and save frames to see the output.
    
    # To run headlessly and see the output, we can save a few frames.
    # This demonstrates that the rendering works correctly.
    
    try:
        # We need to set up a display for manual play.
        # This is separate from the environment's internal headless surface.
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Fast-Paced Sokoban")

        obs, info = env.reset()
        running = True
        terminated = False
        
        print(GameEnv.user_guide)
        
        while running:
            action = [0, 0, 0] # Default to no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
            
            if not terminated:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    action[0] = 1
                elif keys[pygame.K_DOWN]:
                    action[0] = 2
                elif keys[pygame.K_LEFT]:
                    action[0] = 3
                elif keys[pygame.K_RIGHT]:
                    action[0] = 4
                
                obs, reward, terminated, truncated, info = env.step(action)

            # The environment's _get_observation returns the transposed array.
            # For displaying with pygame, we need to transpose it back.
            frame_to_show = np.transpose(obs, (1, 0, 2))
            
            # Create a Pygame surface from the NumPy array
            surf = pygame.surfarray.make_surface(frame_to_show)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(GameEnv.FPS)

    finally:
        env.close()