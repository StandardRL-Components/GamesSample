
# Generated: 2025-08-28T00:00:33.545834
# Source Brief: brief_03658.md
# Brief Index: 3658

        
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
        "Controls: Arrow keys to jump 1 tile. Hold Space + Arrow key for a 2-tile jump. "
        "Space alone jumps 2 tiles in the last direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A survival horror game. Navigate a dark grid, jump to evade creatures, and reach the light gray exit. "
        "Creatures follow set patrol paths."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup (Headless) ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Configuration ---
        self.grid_width = 16
        self.grid_height = 10
        self.cell_size = 40
        self.max_steps = 1000
        self.num_creatures = 5

        # --- Visuals ---
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (35, 40, 55)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (180, 200, 255)
        self.COLOR_EXIT = (180, 180, 180)
        self.COLOR_TEXT = (220, 220, 220)
        self.creature_colors = [
            (255, 60, 60), (255, 100, 100), (230, 40, 90), 
            (255, 80, 40), (240, 70, 70)
        ]

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.player_prev_pos = None
        self.exit_pos = None
        self.creatures = None
        self.last_move_direction_key = 1  # Default to UP
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.near_miss_flash_timer = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [2, self.grid_height // 2]
        self.player_prev_pos = self.player_pos.copy()
        self.exit_pos = [self.grid_width - 3, self.grid_height // 2]
        
        self.creatures = self._generate_creatures()

        self.last_move_direction_key = 1 # 1: up, 2: down, 3: left, 4: right
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.near_miss_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def _generate_creatures(self):
        creatures = []
        occupied_tiles = {tuple(self.player_pos), tuple(self.exit_pos)}
        
        for i in range(self.num_creatures):
            while True:
                route_len = self.np_random.integers(3, 6)
                start_pos = [
                    self.np_random.integers(0, self.grid_width),
                    self.np_random.integers(0, self.grid_height),
                ]

                # Ensure creature doesn't start near player
                if abs(start_pos[0] - self.player_pos[0]) <= 2 and abs(start_pos[1] - self.player_pos[1]) <= 2:
                    continue

                route = self._generate_patrol_route(start_pos, route_len, occupied_tiles)
                if route:
                    creatures.append({
                        "route": route,
                        "patrol_index": 0,
                        "pos": route[0],
                        "id": i
                    })
                    for pos in route:
                        occupied_tiles.add(tuple(pos))
                    break
        return creatures

    def _generate_patrol_route(self, start_pos, length, occupied):
        route = [start_pos]
        current_pos = start_pos
        for _ in range(length - 1):
            possible_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = [current_pos[0] + dx, current_pos[1] + dy]
                if 0 <= next_pos[0] < self.grid_width and \
                   0 <= next_pos[1] < self.grid_height and \
                   tuple(next_pos) not in occupied and \
                   next_pos not in route:
                    possible_moves.append(next_pos)
            
            if not possible_moves:
                return None # Failed to generate a valid route
            
            current_pos = list(self.np_random.choice(possible_moves, axis=0))
            route.append(current_pos)
        return route

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement_key = action[0]
        space_held = action[1] == 1
        # shift_held (action[2]) is unused per design brief

        reward = 0.1  # Survival reward

        # --- Player Movement Logic ---
        self.player_prev_pos = self.player_pos.copy()
        jump_dir_key = 0
        
        if movement_key != 0:
            self.last_move_direction_key = movement_key
            jump_dir_key = movement_key
        elif space_held: # Space w/o direction key
            jump_dir_key = self.last_move_direction_key

        if jump_dir_key != 0:
            dx, dy = 0, 0
            if jump_dir_key == 1: dy = -1   # Up
            elif jump_dir_key == 2: dy = 1  # Down
            elif jump_dir_key == 3: dx = -1 # Left
            elif jump_dir_key == 4: dx = 1  # Right
            
            jump_distance = 2 if space_held else 1
            
            # Check for jumping over a creature
            if jump_distance == 2:
                intermediate_tile = [self.player_pos[0] + dx, self.player_pos[1] + dy]
                for creature in self.creatures:
                    if creature["pos"] == intermediate_tile:
                        reward += 5.0 # Jump over creature reward
                        # SFX: Jump_over_creature.wav

            # Update player position
            self.player_pos[0] += dx * jump_distance
            self.player_pos[1] += dy * jump_distance

            # Clamp to grid
            self.player_pos[0] = max(0, min(self.grid_width - 1, self.player_pos[0]))
            self.player_pos[1] = max(0, min(self.grid_height - 1, self.player_pos[1]))
            
        # --- Creature Movement ---
        for creature in self.creatures:
            creature["patrol_index"] = (creature["patrol_index"] + 1) % len(creature["route"])
            creature["pos"] = creature["route"][creature["patrol_index"]]

        # --- Check Game State & Calculate Rewards ---
        # 1. Collision with creature (Game Over)
        for creature in self.creatures:
            if self.player_pos == creature["pos"]:
                reward = -100.0
                self.game_over = True
                # SFX: Player_death.wav
                break
        
        if not self.game_over:
            # 2. Reached Exit (Game Over)
            if self.player_pos == self.exit_pos:
                reward = 100.0
                self.game_over = True
                # SFX: Level_complete.wav

            # 3. Near Miss
            else:
                is_near_miss = False
                px, py = self.player_pos
                for creature in self.creatures:
                    cx, cy = creature["pos"]
                    if abs(px - cx) <= 1 and abs(py - cy) <= 1:
                        is_near_miss = True
                        break
                if is_near_miss:
                    reward -= 5.0
                    self.near_miss_flash_timer = 3 # frames
                    # SFX: Near_miss_heartbeat.wav

        self.score += reward
        self.steps += 1
        terminated = self.game_over or self.steps >= self.max_steps
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

        # Draw exit
        exit_rect = pygame.Rect(
            self.exit_pos[0] * self.cell_size, self.exit_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)

        # Draw jump trail
        if self.player_prev_pos != self.player_pos:
            trail_rect = pygame.Rect(
                self.player_prev_pos[0] * self.cell_size, self.player_prev_pos[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill((self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], 50))
            self.screen.blit(s, (trail_rect.x, trail_rect.y))

        # Draw creatures
        for creature in self.creatures:
            creature_center_x = int(creature["pos"][0] * self.cell_size + self.cell_size / 2)
            creature_center_y = int(creature["pos"][1] * self.cell_size + self.cell_size / 2)
            color = self.creature_colors[creature["id"] % len(self.creature_colors)]
            size = int(self.cell_size * 0.6)
            
            # Use different shapes for variety
            if creature["id"] % 3 == 0: # Square
                 pygame.draw.rect(self.screen, color, (creature_center_x - size//2, creature_center_y - size//2, size, size))
            elif creature["id"] % 3 == 1: # Circle
                 pygame.gfxdraw.filled_circle(self.screen, creature_center_x, creature_center_y, size // 2, color)
                 pygame.gfxdraw.aacircle(self.screen, creature_center_x, creature_center_y, size // 2, color)
            else: # Triangle
                points = [
                    (creature_center_x, creature_center_y - size//2),
                    (creature_center_x - size//2, creature_center_y + size//2),
                    (creature_center_x + size//2, creature_center_y + size//2),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)


        # Draw player
        player_center_x = int(self.player_pos[0] * self.cell_size + self.cell_size / 2)
        player_center_y = int(self.player_pos[1] * self.cell_size + self.cell_size / 2)
        player_size = int(self.cell_size * 0.7)
        
        # Glow effect
        glow_radius = int(player_size * 1.2)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_center_x - glow_radius, player_center_y - glow_radius))

        # Player square
        player_rect = pygame.Rect(
            player_center_x - player_size / 2, player_center_y - player_size / 2,
            player_size, player_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Near miss flash effect
        if self.near_miss_flash_timer > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            alpha = 90 - (3 - self.near_miss_flash_timer) * 30
            flash_surface.fill((255, 50, 50, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.near_miss_flash_timer -= 1


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.max_steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.screen_width - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Setup Pygame window for human play ---
    try:
        pygame.display.init()
        game_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption("Grid Horror")
        clock = pygame.time.Clock()
        
        terminated = False
        
        print("\n" + "="*30)
        print(env.game_description)
        print(env.user_guide)
        print("="*30 + "\n")
        
        while not terminated:
            # --- Action mapping for human keyboard ---
            movement_key = 0 # 0: none
            space_held = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement_key = 1
            elif keys[pygame.K_DOWN]: movement_key = 2
            elif keys[pygame.K_LEFT]: movement_key = 3
            elif keys[pygame.K_RIGHT]: movement_key = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            
            action = [movement_key, space_held, 0] # Shift is not used
            
            # --- Event handling (e.g., closing the window) ---
            should_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                        should_step = True
                    if event.key == pygame.K_r: # Reset game
                        obs, info = env.reset()
                        print(f"Game Reset. Initial Info: {info}")
                        should_step = False

            if should_step and not env.game_over:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

            # --- Rendering ---
            # The observation is already a rendered frame
            # We just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated and env.game_over:
                print("Game Over! Press 'R' to restart.")

            clock.tick(15) # Limit frame rate for human play

    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Could not create display. This is normal in a headless environment.")
        print("The environment is still valid for training an RL agent.")
    
    env.close()