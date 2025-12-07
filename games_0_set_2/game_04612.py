
# Generated: 2025-08-28T02:55:11.190646
# Source Brief: brief_04612.md
# Brief Index: 4612

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold Space to mine adjacent minerals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spaceship through an asteroid field, mining valuable minerals while avoiding collisions."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

    MAX_STEPS = 1000
    WIN_SCORE = 50
    NUM_ASTEROIDS = 10
    NUM_MINERALS = 60

    # --- Colors ---
    COLOR_BG = (10, 10, 20)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_ASTEROID = (255, 60, 60)
    COLOR_ASTEROID_OUTLINE = (180, 40, 40)
    COLOR_MINERAL = (255, 220, 0)
    COLOR_MINERAL_OUTLINE = (200, 170, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = (0, 0)
        self.asteroids = set()
        self.minerals = set()
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []

        # Generate game elements
        self._place_elements()
        
        return self._get_observation(), self._get_info()

    def _place_elements(self):
        """Generates positions for player, asteroids, and minerals, ensuring no overlaps."""
        all_cells = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        
        # Player in center
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        all_cells.remove(self.player_pos)
        
        # Asteroids
        num_asteroids = min(self.NUM_ASTEROIDS, len(all_cells))
        asteroid_indices = self.np_random.choice(len(all_cells), num_asteroids, replace=False)
        self.asteroids = {list(all_cells)[i] for i in asteroid_indices}
        all_cells -= self.asteroids

        # Minerals
        num_minerals = min(self.NUM_MINERALS, len(all_cells))
        mineral_indices = self.np_random.choice(len(all_cells), num_minerals, replace=False)
        self.minerals = {list(all_cells)[i] for i in mineral_indices}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        self.steps += 1
        reward = 0.1 # Survival reward
        
        # --- 1. Handle Movement ---
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        # Clamp to grid boundaries
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)

        # --- 2. Handle Mining ---
        if space_held:
            mined_this_step = self._attempt_mining()
            if mined_this_step:
                reward += mined_this_step['reward']
                # Spawn a new mineral to replace the mined one
                self._spawn_mineral()

        # --- 3. Check for Collisions ---
        if self.player_pos in self.asteroids:
            self.game_over = True
            self.win = False
            reward = -100.0
            # sfx: player_explosion.wav
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 30)
        
        # --- 4. Check for Win/Loss Conditions ---
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _attempt_mining(self):
        """Checks adjacent cells for minerals and mines one if found."""
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Up, Down, Left, Right
            check_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if check_pos in self.minerals:
                self.minerals.remove(check_pos)
                self.score += 1
                
                # sfx: mine_collect.wav
                self._create_explosion(check_pos, self.COLOR_MINERAL, 15)

                # Check for risky mining bonus
                is_risky = any(
                    abs(check_pos[0] - ax) + abs(check_pos[1] - ay) == 1
                    for ax, ay in self.asteroids
                )
                
                return {"reward": 2.0 if is_risky else 1.0}
        return None

    def _spawn_mineral(self):
        """Spawns a new mineral in a random empty cell."""
        occupied_cells = self.asteroids | self.minerals | {self.player_pos}
        empty_cells = [
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
            if (x, y) not in occupied_cells
        ]
        if empty_cells:
            new_pos = self.np_random.choice(empty_cells)
            self.minerals.add(tuple(new_pos))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._update_and_render_particles()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixels(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        gx, gy = grid_pos
        px = gx * self.GRID_SIZE + self.GRID_SIZE // 2
        py = gy * self.GRID_SIZE + self.GRID_SIZE // 2
        return (px, py)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_entities(self):
        # Render Minerals
        for pos in self.minerals:
            px, py = self._grid_to_pixels(pos)
            size = self.GRID_SIZE // 2
            rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_MINERAL, rect)
            pygame.draw.rect(self.screen, self.COLOR_MINERAL_OUTLINE, rect, 1)

        # Render Asteroids
        for pos in self.asteroids:
            px, py = self._grid_to_pixels(pos)
            radius = int(self.GRID_SIZE * 0.45)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_ASTEROID_OUTLINE)
        
        # Render Player
        if not (self.game_over and not self.win): # Don't draw player if they crashed
            px, py = self._grid_to_pixels(self.player_pos)
            s = self.GRID_SIZE // 2
            points = [(px, py - s * 0.7), (px - s * 0.5, py + s * 0.5), (px + s * 0.5, py + s * 0.5)]
            
            # Glow effect
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
            
            # Main ship
            s = self.GRID_SIZE // 2 - 2
            points = [(px, py - s * 0.7), (px - s * 0.5, py + s * 0.5), (px + s * 0.5, py + s * 0.5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"MINERALS: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_ui.render(f"STEPS: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _create_explosion(self, grid_pos, color, count):
        px, py = self._grid_to_pixels(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": [px, py],
                "vel": velocity,
                "life": self.np_random.integers(15, 30),
                "size": self.np_random.integers(3, 7),
                "color": color,
            }
            self.particles.append(particle)

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.2)
            
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / 30))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(p["size"]), int(p["size"])), int(p["size"]))
                self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")
    
    while not done:
        # --- Human Controls ---
        movement = 0 # No-op
        space = 0 # Released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # Shift is unused
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Game Speed ---
        # Since auto_advance is False, we control the step rate here for human playability
        env.clock.tick(10) # 10 steps per second

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")

    env.close()