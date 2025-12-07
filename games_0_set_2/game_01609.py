
# Generated: 2025-08-28T02:07:52.383656
# Source Brief: brief_01609.md
# Brief Index: 1609

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string, corrected to match the game's mechanics
    user_guide = (
        "Controls: Arrow keys to move. Collect 50 crystals before time runs out!"
    )

    # User-facing description of the game, from the design brief
    game_description = (
        "Navigate a procedurally generated crystal maze, collecting crystals against the clock."
    )

    # Frames advance only when an action is received.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE

        # Game parameters
        self.TIME_LIMIT = 1800  # 60 seconds at 30fps if auto_advance=True, now it's 1800 steps
        self.CRYSTAL_TARGET = 50
        self.TOTAL_CRYSTALS = 75 # More crystals than needed to ensure solvability

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_WALL = (40, 60, 150)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150, 50)
        self.COLOR_CRYSTAL = (0, 255, 255)
        self.COLOR_CRYSTAL_GLOW = (150, 255, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.PARTICLE_COLORS = [(0, 200, 255), (100, 225, 255), (200, 255, 255)]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.walls = set()
        self.crystals = []
        self.particles = []
        self.np_random = None

        # This will call reset() for the first time
        # self.validate_implementation() # Cannot be called here as reset() is needed first

    def _generate_maze(self):
        """Generates a maze using recursive backtracking (DFS)."""
        walls = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if x < self.GRID_WIDTH - 1:
                    walls.add(((x, y), (x + 1, y)))
                if y < self.GRID_HEIGHT - 1:
                    walls.add(((x, y), (x, y + 1)))

        start_node = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
        visited = {start_node}
        stack = [start_node]
        floor_tiles = []

        while stack:
            current_node = stack[-1]
            floor_tiles.append(current_node)
            cx, cy = current_node
            
            neighbors = []
            if cx > 0 and (cx - 1, cy) not in visited: neighbors.append((cx - 1, cy))
            if cx < self.GRID_WIDTH - 1 and (cx + 1, cy) not in visited: neighbors.append((cx + 1, cy))
            if cy > 0 and (cx, cy - 1) not in visited: neighbors.append((cx, cy - 1))
            if cy < self.GRID_HEIGHT - 1 and (cx, cy + 1) not in visited: neighbors.append((cx, cy + 1))

            if neighbors:
                next_node = neighbors[self.np_random.integers(0, len(neighbors))]
                nx, ny = next_node
                
                wall_to_remove = tuple(sorted((current_node, next_node)))
                if wall_to_remove in walls:
                    walls.remove(wall_to_remove)
                
                visited.add(next_node)
                stack.append(next_node)
            else:
                stack.pop()
        
        return walls, list(set(floor_tiles))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Generate maze
        self.walls, floor_tiles = self._generate_maze()
        
        # Place player
        player_start_index = self.np_random.integers(0, len(floor_tiles))
        self.player_pos = pygame.Vector2(floor_tiles.pop(player_start_index))

        # Place crystals
        self.crystals = []
        crystal_indices = self.np_random.choice(len(floor_tiles), self.TOTAL_CRYSTALS, replace=False)
        for i in crystal_indices:
            self.crystals.append(pygame.Vector2(floor_tiles[i]))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self._handle_movement(movement)
        
        # Update particles
        self._update_particles()

        self.steps += 1
        reward = self._check_crystal_collection()
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.CRYSTAL_TARGET:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        current_pos = (int(self.player_pos.x), int(self.player_pos.y))
        target_pos = pygame.Vector2(self.player_pos)

        if movement == 1: # Up
            target_pos.y -= 1
        elif movement == 2: # Down
            target_pos.y += 1
        elif movement == 3: # Left
            target_pos.x -= 1
        elif movement == 4: # Right
            target_pos.x += 1
        else: # No-op
            return

        # Boundary check
        if not (0 <= target_pos.x < self.GRID_WIDTH and 0 <= target_pos.y < self.GRID_HEIGHT):
            return

        # Wall collision check
        next_pos = (int(target_pos.x), int(target_pos.y))
        wall_to_check = tuple(sorted((current_pos, next_pos)))
        if wall_to_check not in self.walls:
            self.player_pos = target_pos

    def _check_crystal_collection(self):
        for crystal in self.crystals:
            if self.player_pos == crystal:
                self.crystals.remove(crystal)
                self.score += 1
                self._spawn_particles(crystal)
                # SFX: Crystal collect sound
                return 1 # Reward for collecting a crystal
        return 0

    def _check_termination(self):
        if self.score >= self.CRYSTAL_TARGET:
            return True
        if self.steps >= self.TIME_LIMIT:
            return True
        return False

    def _spawn_particles(self, pos):
        center_x = (pos.x + 0.5) * self.CELL_SIZE
        center_y = (pos.y + 0.5) * self.CELL_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            particle = {
                'pos': pygame.Vector2(center_x, center_y),
                'vel': velocity,
                'life': self.np_random.uniform(10, 20),
                'color': random.choice(self.PARTICLE_COLORS),
                'radius': self.np_random.uniform(2, 4)
            }
            self.particles.append(particle)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            p['radius'] -= 0.1
    
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
        # Draw walls
        for wall in self.walls:
            p1 = ( (wall[0][0] + 0.5) * self.CELL_SIZE, (wall[0][1] + 0.5) * self.CELL_SIZE )
            p2 = ( (wall[1][0] + 0.5) * self.CELL_SIZE, (wall[1][1] + 0.5) * self.CELL_SIZE )
            pygame.draw.line(self.screen, self.COLOR_WALL, p1, p2, 3)

        # Draw crystals with sparkle effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        for crystal in self.crystals:
            center_x = int((crystal.x + 0.5) * self.CELL_SIZE)
            center_y = int((crystal.y + 0.5) * self.CELL_SIZE)
            
            # Pulsing glow
            glow_radius = int(self.CELL_SIZE * 0.4 + pulse * 4)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_CRYSTAL_GLOW)
            
            # Crystal polygon
            radius = self.CELL_SIZE * 0.3
            points = []
            for i in range(6):
                angle = (i * 2 * math.pi / 6) + (self.steps * 0.05)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['radius'])))

        # Draw player
        player_center_x = int((self.player_pos.x + 0.5) * self.CELL_SIZE)
        player_center_y = int((self.player_pos.y + 0.5) * self.CELL_SIZE)
        player_size = int(self.CELL_SIZE * 0.8)
        player_rect = pygame.Rect(0, 0, player_size, player_size)
        player_rect.center = (player_center_x, player_center_y)
        
        # Player glow
        glow_size = int(player_size * 1.5)
        pygame.gfxdraw.box(self.screen, player_rect.inflate(glow_size-player_size, glow_size-player_size), self.COLOR_PLAYER_GLOW)

        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)


    def _render_ui(self):
        # Score display
        score_text = f"CRYSTALS: {self.score} / {self.CRYSTAL_TARGET}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer display
        time_left = max(0, self.TIME_LIMIT - self.steps)
        time_text = f"TIME: {time_left}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_surf, time_rect)

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.CRYSTAL_TARGET:
                end_text = "YOU WIN!"
            else:
                end_text = "TIME'S UP!"
            
            end_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.TIME_LIMIT - self.steps),
            "crystals_left": len(self.crystals)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        Note: Must be called after the first reset() to initialize state.
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set up Pygame for human interaction
    pygame.display.set_caption("Crystal Maze")
    screen = pygame.display.set_mode((640, 400))
    
    env = GameEnv(render_mode="rgb_array")
    
    # Run validation after the first reset
    obs, info = env.reset(seed=42)
    env.validate_implementation()
    
    terminated = False
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
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
            
            # Since auto_advance is False, we must step even on no-op to process key releases
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate for human playability
        env.clock.tick(15)

    env.close()