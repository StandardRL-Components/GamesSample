
# Generated: 2025-08-27T13:35:59.329414
# Source Brief: brief_00424.md
# Brief Index: 424

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move or attack. Each action is one turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A roguelike dungeon crawler. Navigate the grid, defeat enemies by bumping into them, "
        "collect gold, and reach the blue exit tile to win. Moving next to an enemy will cause you to take damage."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.UI_HEIGHT = 40
        self.TILE_SIZE = (self.HEIGHT - self.UI_HEIGHT) // self.GRID_SIZE
        self.GAME_AREA_WIDTH = self.TILE_SIZE * self.GRID_SIZE
        self.GAME_AREA_OFFSET_X = (self.WIDTH - self.GAME_AREA_WIDTH) // 2
        self.GAME_AREA_OFFSET_Y = self.UI_HEIGHT

        self.MAX_STEPS = 1000
        self.START_HEALTH = 5
        self.NUM_ENEMIES = 3
        self.NUM_GOLD = 5

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_UI_BG = (30, 30, 40)
        self.COLOR_FLOOR = (50, 50, 60)
        self.COLOR_GRID = (70, 70, 80)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_EXIT = (50, 100, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HIT_FLASH = (255, 255, 255)
        self.COLOR_ENEMY_HIT_FLASH = (255, 150, 150)

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
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.gold = 0
        self.player_health = 0
        self.player_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.enemies = []
        self.golds = []
        self.game_over = False
        
        # Visual effect trackers
        self.player_hit_flash_timer = 0
        self.enemy_hit_effects = [] # Stores [pos, timer]

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.gold = 0
        self.player_health = self.START_HEALTH
        self.game_over = False

        self.player_hit_flash_timer = 0
        self.enemy_hit_effects = []
        
        # Procedurally generate the level
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        reward = 0.0

        # --- Handle Action ---
        if movement != 0: # 0 is no-op
            old_pos = list(self.player_pos)
            old_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            target_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            # Check for wall collision
            if not (0 <= target_pos[0] < self.GRID_SIZE and 0 <= target_pos[1] < self.GRID_SIZE):
                # Bumped into a wall, no action, no reward change
                pass
            
            # Check for enemy collision (attack)
            elif target_pos in self.enemies:
                # Sound: player_attack.wav
                self.enemies.remove(target_pos)
                self._spawn_enemy()
                reward += 1.0  # Reward for defeating an enemy
                self.score += 1.0
                self.enemy_hit_effects.append([list(target_pos), 10]) # Add flash effect
            
            # Move player
            else:
                self.player_pos = target_pos
                
                # Check for gold collection
                if self.player_pos in self.golds:
                    # Sound: collect_gold.wav
                    self.golds.remove(self.player_pos)
                    self.gold += 1
                    reward += 0.5
                    self.score += 0.5

                # Check for enemy proximity damage
                for enemy_pos in self.enemies:
                    if self._manhattan_distance(self.player_pos, enemy_pos) == 1:
                        # Sound: player_hit.wav
                        self.player_health -= 1
                        self.player_hit_flash_timer = 10 # frames
                        break # Only take damage once per turn
                
                # Reward for moving closer/further from exit
                new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
                if new_dist_to_exit < old_dist_to_exit:
                    reward += 0.1
                    self.score += 0.1
                elif new_dist_to_exit > old_dist_to_exit:
                    reward -= 0.1
                    self.score -= 0.1

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos == self.exit_pos:
            # Sound: victory.wav
            reward += 100.0
            self.score += 100.0
            terminated = True
            self.game_over = True
        elif self.player_health <= 0:
            # Sound: game_over.wav
            reward -= 100.0
            self.score -= 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # Update visual effects
        self._update_effects()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_level(self):
        """Creates a new level layout."""
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)

        # Place player
        self.player_pos = list(all_coords.pop())

        # Define area around player to keep clear initially
        player_neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                if neighbor in all_coords:
                    all_coords.remove(neighbor)
        
        # Place exit
        self.exit_pos = list(all_coords.pop())

        # Place enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            if all_coords:
                self.enemies.append(list(all_coords.pop()))

        # Place gold
        self.golds = []
        for _ in range(self.NUM_GOLD):
            if all_coords:
                self.golds.append(list(all_coords.pop()))

    def _spawn_enemy(self):
        """Spawns a single enemy in a valid, unoccupied location."""
        occupied = [self.player_pos, self.exit_pos] + self.enemies + self.golds
        possible_coords = []
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if [x, y] not in occupied:
                    possible_coords.append([x, y])
        
        if possible_coords:
            self.enemies.append(random.choice(possible_coords))


    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _update_effects(self):
        """Update timers for visual effects."""
        if self.player_hit_flash_timer > 0:
            self.player_hit_flash_timer -= 1
        
        self.enemy_hit_effects = [[pos, timer - 1] for pos, timer in self.enemy_hit_effects if timer > 1]

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
        # Draw grid floor
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GAME_AREA_OFFSET_X + c * self.TILE_SIZE,
                    self.GAME_AREA_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # --- Draw Game Entities ---
        # Exit
        self._draw_tile_entity(self.exit_pos, self.COLOR_EXIT, 'exit')
        # Gold
        for pos in self.golds:
            self._draw_tile_entity(pos, self.COLOR_GOLD, 'gold')
        # Enemies
        for pos in self.enemies:
            self._draw_tile_entity(pos, self.COLOR_ENEMY, 'enemy')
        # Player
        self._draw_tile_entity(self.player_pos, self.COLOR_PLAYER, 'player')

        # --- Draw Effects ---
        # Player hit flash
        if self.player_hit_flash_timer > 0:
            self._draw_tile_entity(self.player_pos, self.COLOR_HIT_FLASH, 'flash')
        # Enemy hit flash
        for pos, timer in self.enemy_hit_effects:
             self._draw_tile_entity(pos, self.COLOR_ENEMY_HIT_FLASH, 'flash')

    def _draw_tile_entity(self, pos, color, entity_type):
        """Helper to draw an entity on the grid."""
        center_x = self.GAME_AREA_OFFSET_X + int((pos[0] + 0.5) * self.TILE_SIZE)
        center_y = self.GAME_AREA_OFFSET_Y + int((pos[1] + 0.5) * self.TILE_SIZE)
        radius = int(self.TILE_SIZE * 0.35)

        if entity_type == 'player':
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        elif entity_type == 'enemy':
            points = [
                (center_x, center_y - radius),
                (center_x + radius, center_y + radius),
                (center_x - radius, center_y + radius),
            ]
            pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], color)
            pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], color)
        elif entity_type == 'gold':
            size = int(self.TILE_SIZE * 0.6)
            rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif entity_type == 'exit':
            size = int(self.TILE_SIZE * 0.8)
            rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
        elif entity_type == 'flash':
             size = int(self.TILE_SIZE)
             rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
             flash_surface = pygame.Surface((size, size), pygame.SRCALPHA)
             flash_surface.fill((*color, 128))
             self.screen.blit(flash_surface, rect.topleft)


    def _render_ui(self):
        # Draw UI background
        ui_rect = pygame.Rect(0, 0, self.WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT - 1), (self.WIDTH, self.UI_HEIGHT - 1))

        # Health display
        health_text = self.font_main.render(f"Health: {self.player_health}/{self.START_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 8))
        
        # Gold display
        gold_text = self.font_main.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (150, 8))

        # Steps display
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - 150, 8))

    def _get_info(self):
        return {
            "score": self.gold,
            "steps": self.steps,
            "health": self.player_health
        }
        
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Roguelike Dungeon")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default action: no-op, buttons released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    terminated = False
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit
                    running = False
        
        # In manual play, we only step when an action is taken
        if action[0] != 0 and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            font = pygame.font.Font(None, 74)
            text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 36)
            text_small = font_small.render("Press 'R' to restart or 'Q' to quit", True, (255, 255, 255))
            text_small_rect = text_small.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 50))
            screen.blit(text_small, text_small_rect)
            
            pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()