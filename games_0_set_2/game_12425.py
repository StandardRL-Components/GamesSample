import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.lifespan = random.randint(20, 40)
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.1
        self.vx *= 0.95
        self.vy *= 0.95

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent navigates a grid to find hidden apples.
    The agent must strategically reveal tiles to locate apples and collect them
    before a step limit is reached. The core challenge is balancing exploration
    (revealing new tiles) with exploitation (collecting known apples).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a grid to find and collect hidden apples. "
        "Strategically reveal tiles to locate all apples before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to reveal the tile in front of you "
        "and shift to reveal the tile to your right."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.TILE_SIZE = 20
        self.MAX_STEPS = 500
        self.NUM_APPLES = 10
        self.INITIAL_REVEAL_PERCENT = 0.20

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_LINES = (40, 45, 50)
        self.COLOR_HIDDEN = (60, 65, 70)
        self.COLOR_REVEALED = (120, 125, 130)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_APPLE = (255, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        if self.render_mode == "human":
            pygame.display.set_caption("Tile Reveal Gridworld")
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            self.window = None
        
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_dir = (0, -1) # (dx, dy) for "front" direction
        self.apples = set()
        self.revealed_tiles = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        
        # --- Generate Game Board ---
        self.revealed_tiles.fill(False)
        self.apples.clear()
        
        # Generate apple positions
        while len(self.apples) < self.NUM_APPLES:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            self.apples.add(pos)
        
        # Reveal initial tiles, guaranteeing one apple is visible
        tiles_to_reveal = set()
        visible_apple_pos = random.choice(list(self.apples))
        tiles_to_reveal.add(visible_apple_pos)
        
        num_initial_reveals = int(self.GRID_WIDTH * self.GRID_HEIGHT * self.INITIAL_REVEAL_PERCENT)
        while len(tiles_to_reveal) < num_initial_reveals:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            tiles_to_reveal.add(pos)
        
        for (x, y) in tiles_to_reveal:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.revealed_tiles[y, x] = True
                
        # Place player on a random, revealed, non-apple tile
        possible_starts = [pos for pos in tiles_to_reveal if pos not in self.apples]
        if not possible_starts: # Failsafe if all revealed tiles have apples
            possible_starts = [(x,y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT) if (x,y) not in self.apples]
        
        self.player_pos = list(random.choice(possible_starts))
        self.player_dir = (0, -1) # Default facing up
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.player_dir = (dx, dy)
            new_x = np.clip(self.player_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            new_y = np.clip(self.player_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
            self.player_pos = [new_x, new_y]
        
        # Reveal tile in front (on press)
        if space_held and not self.prev_space_held:
            front_x = self.player_pos[0] + self.player_dir[0]
            front_y = self.player_pos[1] + self.player_dir[1]
            if 0 <= front_x < self.GRID_WIDTH and 0 <= front_y < self.GRID_HEIGHT:
                if not self.revealed_tiles[front_y, front_x]:
                    self.revealed_tiles[front_y, front_x] = True
                    if (front_x, front_y) in self.apples:
                        reward += 0.1 # Reward for revealing an apple
        
        # Reveal tile to the right (on press)
        if shift_held and not self.prev_shift_held:
            right_dir = (self.player_dir[1], -self.player_dir[0]) # Rotate 90 deg clockwise
            right_x = self.player_pos[0] + right_dir[0]
            right_y = self.player_pos[1] + right_dir[1]
            if 0 <= right_x < self.GRID_WIDTH and 0 <= right_y < self.GRID_HEIGHT:
                if not self.revealed_tiles[right_y, right_x]:
                    self.revealed_tiles[right_y, right_x] = True
                    if (right_x, right_y) in self.apples:
                        reward += 0.1 # Reward for revealing an apple

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        
        # Check for apple collection
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.apples:
            self.apples.remove(player_pos_tuple)
            self.score += 1
            reward += 1.0
            self._spawn_particles(
                (self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2,
                 self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2),
                self.COLOR_APPLE
            )

        # --- Termination ---
        terminated = self.steps >= self.MAX_STEPS or len(self.apples) == 0
        if terminated and len(self.apples) == 0:
            reward += 10.0 # Bonus for collecting all apples
        
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_to_window()

        return observation, reward, terminated, False, info

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        for p in reversed(self.particles):
            p.update()
            if p.lifespan <= 0 or p.radius <= 0:
                self.particles.remove(p)
            else:
                p.draw(self.screen)

        # Draw grid and tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                
                if self.revealed_tiles[y, x]:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    if (x, y) in self.apples:
                        apple_center = (rect.centerx, rect.centery)
                        pygame.draw.circle(self.screen, self.COLOR_APPLE, apple_center, self.TILE_SIZE // 3)
                else:
                    # Subtle flicker for hidden tiles
                    flicker = self.np_random.integers(-5, 6)
                    flicker_color = tuple(np.clip([c + flicker for c in self.COLOR_HIDDEN], 0, 255))
                    pygame.draw.rect(self.screen, flicker_color, rect)

                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

        # Draw player
        player_center_x = int(self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        player_center_y = int(self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        
        # Glow effect using a temporary surface for alpha blending
        glow_radius = int(self.TILE_SIZE * 0.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_center_x - glow_radius, player_center_y - glow_radius))

        # Player body
        player_radius = int(self.TILE_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        
        # Player direction indicator
        dir_end_x = player_center_x + self.player_dir[0] * player_radius * 0.8
        dir_end_y = player_center_y + self.player_dir[1] * player_radius * 0.8
        pygame.draw.line(self.screen, self.COLOR_BG, (player_center_x, player_center_y), (int(dir_end_x), int(dir_end_y)), 2)


    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer display
        time_left = self.MAX_STEPS - self.steps
        time_color = self.COLOR_TEXT if time_left > 50 else self.COLOR_APPLE
        time_text = self.font_main.render(f"Time: {time_left}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _render_to_window(self):
        if self.window is not None:
            # The observation is already rendered to self.screen
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "apples_left": len(self.apples),
            "player_pos": tuple(self.player_pos)
        }

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            self.particles.append(Particle(pos[0], pos[1], color))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game with keyboard controls for testing.
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
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

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            pygame.time.wait(1000)

    env.close()