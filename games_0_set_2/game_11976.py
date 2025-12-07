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
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a dynamic maze survival game.

    The player must navigate a maze with oscillating walls to reach an exit,
    while avoiding enemies that spawn and move towards the player.

    **Visuals:**
    - Player: Bright blue square with a glow.
    - Enemies: Bright red squares.
    - Exit: Bright yellow square.
    - Walls: Oscillate between green (open) and red (closed).
    - Background: A dark grid.

    **Gameplay:**
    - The game is turn-based. Each `step()` constitutes one turn.
    - In each turn: player moves, walls might change state, enemies move, new enemies spawn.
    - The episode ends if the player reaches the exit (win), collides with an enemy (loss),
      or the step limit is reached.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `action[1]`: Unused (space button)
    - `action[2]`: Unused (shift button)

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - +1000 for reaching the exit.
    - -50 for colliding with an enemy.
    - +1 for every step survived otherwise.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a maze with oscillating walls to reach the exit while avoiding enemies."
    user_guide = "Controls: Use ↑↓←→ arrow keys to move. Reach the yellow exit and avoid red enemies."
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 12
        self.CELL_SIZE = 32  # 640/20=32, 400/12.5 -> use 12 height -> 384px, with margin
        self.MARGIN_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        self.MAX_STEPS = 1000
        self.WALL_TOGGLE_INTERVAL = 30
        self.MAX_ENEMY_SPAWN_PER_STEP = 3
        self.MIN_ENEMY_SPAWN_PER_STEP = 1

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 24)
            self.font_large = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_EXIT = (255, 220, 0)
        self.COLOR_WALL_CLOSED = (200, 60, 60)
        self.COLOR_WALL_OPEN = (60, 200, 60)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.enemies = []
        self.wall_locations = []
        self.wall_state_timer = 0
        self.walls_are_closed = False
        self.game_outcome = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # --- Initialize Game Elements ---
        # Player and Exit
        self.player_pos = [1, 1]
        self.exit_pos = [self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2]

        # Enemies
        self.enemies = []

        # Walls - create a static checkerboard pattern of potential walls
        if not self.wall_locations:
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    is_border = c == 0 or c == self.GRID_WIDTH - 1 or r == 0 or r == self.GRID_HEIGHT - 1
                    is_near_start = abs(c - self.player_pos[0]) <= 1 and abs(r - self.player_pos[1]) <= 1
                    is_near_exit = abs(c - self.exit_pos[0]) <= 1 and abs(r - self.exit_pos[1]) <= 1
                    if (c + r) % 2 != 0 and not is_border and not is_near_start and not is_near_exit:
                        self.wall_locations.append([c, r])
        
        self.wall_state_timer = 0
        self.walls_are_closed = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = 0
        
        # 1. Update wall state
        self._update_walls()

        # 2. Update player position based on action
        self._update_player(movement)

        # 3. Update enemies (move, despawn if hitting a wall)
        self._update_enemies()

        # 4. Spawn new enemies
        self._spawn_enemies()

        # 5. Check for terminal conditions (win/loss)
        is_victory = tuple(self.player_pos) == tuple(self.exit_pos)
        is_collision = any(tuple(self.player_pos) == tuple(e) for e in self.enemies)
        is_timeout = self.steps >= self.MAX_STEPS

        terminated = is_victory or is_collision
        truncated = is_timeout

        # 6. Calculate reward
        if is_victory:
            reward = 1000
            self.score += 1000
            self.game_over = True
            self.game_outcome = "VICTORY!"
        elif is_collision:
            reward = -50
            self.score -= 50
            self.game_over = True
            self.game_outcome = "GAME OVER"
        elif is_timeout:
            reward = 0 # No penalty or reward for timeout
            self.game_over = True
            self.game_outcome = "TIME OUT"
        else:
            reward = 1.0  # Survival reward
            self.score += 1

        if self.game_over:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _is_wall(self, pos):
        """Check if a given grid coordinate is a currently active wall."""
        return self.walls_are_closed and list(pos) in self.wall_locations

    def _update_walls(self):
        """Update the open/closed state of walls based on a timer."""
        self.wall_state_timer += 1
        if self.wall_state_timer >= self.WALL_TOGGLE_INTERVAL:
            self.wall_state_timer = 0
            self.walls_are_closed = not self.walls_are_closed

    def _update_player(self, movement):
        """Move the player based on the movement action, checking for boundaries and walls."""
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx == 0 and dy == 0:
            return

        next_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        # Boundary checks
        if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
            return

        # Wall collision check
        if self._is_wall(next_pos):
            return

        self.player_pos = next_pos

    def _update_enemies(self):
        """Move all enemies towards the player and despawn any that hit a wall."""
        # Move enemies
        for enemy in self.enemies:
            px, py = self.player_pos
            ex, ey = enemy
            
            # Simple pathfinding: move horizontally first, then vertically
            if ex < px: enemy[0] += 1
            elif ex > px: enemy[0] -= 1
            elif ey < py: enemy[1] += 1
            elif ey > py: enemy[1] -= 1

        # Despawn enemies that moved into a wall
        self.enemies = [e for e in self.enemies if not self._is_wall(e)]

    def _spawn_enemies(self):
        """Spawn a random number of new enemies on valid empty cells."""
        num_to_spawn = self.np_random.integers(self.MIN_ENEMY_SPAWN_PER_STEP, self.MAX_ENEMY_SPAWN_PER_STEP + 1)
        
        # Create a set of all occupied cells for efficient lookup
        occupied_cells = {tuple(self.player_pos), tuple(self.exit_pos)}
        for e in self.enemies:
            occupied_cells.add(tuple(e))
        if self.walls_are_closed:
            for w in self.wall_locations:
                occupied_cells.add(tuple(w))

        # Create a list of all possible valid spawn locations
        valid_spawns = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (c, r) not in occupied_cells:
                    valid_spawns.append([c, r])
        
        # Shuffle to randomize spawn choices
        if valid_spawns:
             self.np_random.shuffle(valid_spawns)

        for _ in range(num_to_spawn):
            if not valid_spawns:
                break # No space left to spawn
            
            new_enemy_pos = valid_spawns.pop()
            self.enemies.append(new_enemy_pos)

    def _get_observation(self):
        """Render the current game state to an RGB array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Render all game elements like grid, walls, player, enemies, exit."""
        # Draw grid
        for r in range(self.GRID_HEIGHT + 1):
            y = self.MARGIN_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        for c in range(self.GRID_WIDTH + 1):
            x = c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.MARGIN_Y), (x, self.SCREEN_HEIGHT - self.MARGIN_Y))

        # Draw walls
        wall_color = self.COLOR_WALL_CLOSED if self.walls_are_closed else self.COLOR_WALL_OPEN
        for pos in self.wall_locations:
            rect = self._get_rect(pos)
            pygame.draw.rect(self.screen, wall_color, rect)

        # Draw exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self._get_rect(self.exit_pos))

        # Draw enemies
        for pos in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, self._get_rect(pos))

        # Draw player with glow
        player_rect = self._get_rect(self.player_pos)
        glow_rect = player_rect.inflate(self.CELL_SIZE * 0.5, self.CELL_SIZE * 0.5)
        
        # Create a temporary surface for the glow effect
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surface,
            (*self.COLOR_PLAYER_GLOW, 60), # Use alpha for transparency
            (glow_rect.width // 2, glow_rect.height // 2),
            glow_rect.width // 2
        )
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        """Render UI elements like score and game over text."""
        # Draw score
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, (12, 12), self.font, self.COLOR_TEXT, shadow=True)

        # Draw wall timer
        timer_percent = self.wall_state_timer / self.WALL_TOGGLE_INTERVAL
        timer_width = 100
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.SCREEN_WIDTH - timer_width - 10, 15, timer_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_WALL_OPEN, (self.SCREEN_WIDTH - timer_width - 10, 15, timer_width * timer_percent, 10))
        
        # Draw game over screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.game_outcome, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, self.COLOR_TEXT, shadow=True, center=True)
            
    def _draw_text(self, text, pos, font, color, shadow=False, center=False):
        """Helper function to draw text with optional shadow and centering."""
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect()
            shadow_rect.topleft = (text_rect.left + 2, text_rect.top + 2)
            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

    def _get_rect(self, grid_pos):
        """Convert grid coordinates to a pygame.Rect for drawing."""
        x = grid_pos[0] * self.CELL_SIZE
        y = self.MARGIN_Y + grid_pos[1] * self.CELL_SIZE
        return pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

    def _get_info(self):
        """Return a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "enemy_count": len(self.enemies),
            "walls_closed": self.walls_are_closed,
        }

    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over = False
    
    # Create a display screen separate from the environment's internal surface
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Survival - Manual Test")
    
    action = [0, 0, 0] # Start with no-op

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_over = False
                
                # Update movement part of the action
                if not game_over:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    else: action[0] = 0
                else:
                    action[0] = 0

        # --- Game Step ---
        if not game_over:
            # Only step if a movement key was pressed or if it's a no-op game
            # Since auto_advance is False, we only step on explicit actions.
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                game_over = terminated or truncated
                action[0] = 0 # Reset action to no-op after one move
            else: # Render even if no action is taken
                 obs = env._get_observation()
        
        # --- Rendering ---
        # The observation is the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(10) # Control the speed of manual play

    env.close()