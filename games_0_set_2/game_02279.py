
# Generated: 2025-08-28T04:18:40.236611
# Source Brief: brief_02279.md
# Brief Index: 2279

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An 8x8 grid-based puzzle/adventure game where the player collects crystals
    while avoiding traps. The game is presented in a visually polished interface
    with glowing elements and particle effects.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move your avatar. "
        "Collect 10 crystals to win, but avoid triggering 3 traps!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a procedurally generated crystal cavern, collecting crystals while "
        "avoiding deadly traps. Each move requires careful planning and risk assessment."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_CRYSTAL = (255, 255, 100)
    COLOR_CRYSTAL_GLOW = (128, 128, 50)
    COLOR_TRAP = (100, 0, 0)
    COLOR_TRAP_ARMED = (180, 0, 0)
    COLOR_TRAP_TRIGGERED = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Grid and Screen
    GRID_SIZE = 8
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_SIZE = 320  # 320x320 grid area
    CELL_SIZE = GRID_AREA_SIZE // GRID_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_SIZE) // 2

    # Game Rules
    CRYSTAL_TARGET = 10
    TRAP_LIMIT = 3
    MAX_STEPS = 1000
    NUM_TRAPS = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.player_pos = None
        self.last_player_pos = None
        self.crystals = None
        self.traps = None
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.traps_triggered = 0
        self.game_over = False
        self.win_condition = False
        
        # Animation and effects
        self.animations = []
        self.particles = []
        self.animation_counter = 0

        self.reset()
        
        # This check is for development and ensures API compliance
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.traps_triggered = 0
        self.game_over = False
        self.win_condition = False
        self.animations.clear()
        self.particles.clear()
        self.animation_counter = 0

        self._generate_level()
        self.last_player_pos = self.player_pos

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        while True:
            # 1. Place player
            self.player_pos = (
                self.np_random.integers(0, self.GRID_SIZE),
                self.np_random.integers(0, self.GRID_SIZE)
            )

            # 2. Place traps
            all_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            all_cells.remove(self.player_pos)
            self.np_random.shuffle(all_cells)
            self.traps = set(all_cells[:self.NUM_TRAPS])

            # 3. Check if start position is valid (at least 2 safe moves)
            safe_moves = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in self.traps:
                    safe_moves += 1
            if safe_moves < 2:
                continue # Regenerate level

            # 4. Find all reachable safe cells via BFS
            queue = deque([self.player_pos])
            reachable_safe_cells = {self.player_pos}
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in self.traps and (nx, ny) not in reachable_safe_cells:
                        reachable_safe_cells.add((nx, ny))
                        queue.append((nx, ny))
            
            # 5. Place crystals in reachable safe cells
            potential_crystal_locs = list(reachable_safe_cells - {self.player_pos})
            if len(potential_crystal_locs) >= self.CRYSTAL_TARGET:
                self.np_random.shuffle(potential_crystal_locs)
                self.crystals = set(potential_crystal_locs[:self.CRYSTAL_TARGET])
                break # Successful generation

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.last_player_pos = self.player_pos
        
        # --- Update Game Logic ---
        reward = 0
        moved = False
        if movement != 0: # 0 is no-op
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player_pos = new_pos
                moved = True
        
        if not moved:
            reward -= 0.1 # Small penalty for no-op or hitting a wall
            
        # --- Calculate Reward & Handle Events ---
        reward += self._calculate_reward()
        
        # Check for crystal collection
        if self.player_pos in self.crystals:
            self.crystals.remove(self.player_pos)
            self.crystals_collected += 1
            reward += 10.0
            self.score += 10
            self._create_particles(self.player_pos, self.COLOR_CRYSTAL, 20)
            # SFX: Crystal collect sound

        # Check for trap trigger
        if self.player_pos in self.traps:
            self.traps_triggered += 1
            reward -= 30.0
            self.score -= 30
            self.animations.append({
                "pos": self.player_pos, "type": "trap_trigger", "timer": 20
            })
            # SFX: Trap spring sound
            # Trap is not removed, can be triggered again

        self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_condition:
                reward += 100
                self.score += 100
            else:
                reward -= 100
                self.score -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_reward(self):
        # Distance-based shaping reward
        shaping_reward = 0
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Reward for moving closer to the nearest crystal
        if self.crystals:
            dist_before = min(manhattan_distance(self.last_player_pos, c) for c in self.crystals)
            dist_after = min(manhattan_distance(self.player_pos, c) for c in self.crystals)
            if dist_after < dist_before:
                shaping_reward += 1.0
            elif dist_after > dist_before:
                shaping_reward -= 1.0

        # Penalty for moving closer to the nearest trap
        if self.traps:
            dist_before = min(manhattan_distance(self.last_player_pos, t) for t in self.traps)
            dist_after = min(manhattan_distance(self.player_pos, t) for t in self.traps)
            if dist_after < dist_before:
                shaping_reward -= 1.0
            # No reward for moving away from trap, to avoid encouraging passivity
        
        return shaping_reward

    def _check_termination(self):
        if self.crystals_collected >= self.CRYSTAL_TARGET:
            self.win_condition = True
            return True
        if self.traps_triggered >= self.TRAP_LIMIT:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.animation_counter += 1
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_collected": self.crystals_collected,
            "traps_remaining": self.TRAP_LIMIT - self.traps_triggered,
        }
        
    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_AREA_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw traps
        for trap_pos in self.traps:
            px, py = self._grid_to_pixel(trap_pos)
            is_armed = self.player_pos != trap_pos
            color = self.COLOR_TRAP_ARMED if is_armed else self.COLOR_TRAP
            size = self.CELL_SIZE // 3
            pygame.draw.rect(self.screen, color, (px - size//2, py - size//2, size, size), 2)

        # Draw crystals
        for crystal_pos in self.crystals:
            px, py = self._grid_to_pixel(crystal_pos)
            pulse = (math.sin(self.animation_counter * 0.1) + 1) / 2 # 0 to 1
            
            # Glow
            glow_radius = int(self.CELL_SIZE * 0.5 * (1 + pulse * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_CRYSTAL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_CRYSTAL_GLOW)
            
            # Core
            size = int(self.CELL_SIZE * 0.3 * (1 + pulse * 0.2))
            points = [
                (px, py - size), (px + size, py),
                (px, py + size), (px - size, py)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)

        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        pulse = (math.sin(self.animation_counter * 0.2) + 1) / 2
        
        # Glow
        glow_radius = int(self.CELL_SIZE * 0.4 * (1 + pulse * 0.3))
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Core
        player_size = self.CELL_SIZE // 2
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px - player_size // 2, py - player_size // 2, player_size, player_size), border_radius=3)
        
        # Update and draw animations
        self._update_and_draw_effects()

    def _update_and_draw_effects(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['timer'] -= 1
            if p['timer'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, 255 * (p['timer'] / p['max_timer']))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))
        
        # Update and draw animations
        for anim in self.animations[:]:
            anim['timer'] -= 1
            if anim['timer'] <= 0:
                self.animations.remove(anim)
            else:
                if anim['type'] == 'trap_trigger':
                    px, py = self._grid_to_pixel(anim['pos'])
                    flash_alpha = (math.sin(anim['timer'] * 0.8) + 1) / 2
                    radius = self.CELL_SIZE * 0.7 * (1 - (anim['timer'] / 20))
                    color = (*self.COLOR_TRAP_TRIGGERED, int(255 * flash_alpha))
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius), color)
                    self.screen.blit(temp_surf, (int(px - radius), int(py - radius)))

    def _create_particles(self, grid_pos, color, count):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            timer = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'timer': timer,
                'max_timer': timer,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            img = font.render(text, True, color)
            self.screen.blit(img, pos)

        # Crystal UI
        crystal_icon_pos = (20, 20)
        pygame.gfxdraw.aapolygon(self.screen, [(28, 20), (36, 28), (28, 36), (20, 28)], self.COLOR_CRYSTAL)
        draw_text(f"{self.crystals_collected} / {self.CRYSTAL_TARGET}", self.font_small, self.COLOR_TEXT, (45, 22))

        # Traps UI
        lives_text = f"{self.TRAP_LIMIT - self.traps_triggered} Lives"
        text_w = self.font_small.size(lives_text)[0]
        trap_icon_pos = (self.SCREEN_WIDTH - 30 - text_w, 20)
        pygame.draw.rect(self.screen, self.COLOR_TRAP_ARMED, (trap_icon_pos[0], trap_icon_pos[1], 16, 16), 2)
        draw_text(lives_text, self.font_small, self.COLOR_TEXT, (trap_icon_pos[0] + 25, 22))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = self.COLOR_CRYSTAL if self.win_condition else self.COLOR_TRAP_TRIGGERED
            
            text_w, text_h = self.font_large.size(message)
            draw_text(message, self.font_large, color, 
                      ((self.SCREEN_WIDTH - text_w) // 2, (self.SCREEN_HEIGHT - text_h) // 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up a window to see the game
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action is no-op
        
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
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q:
                    running = False
                    
                # We only step on a key press since auto_advance is False
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

                    if terminated:
                        print("--- Episode Finished ---")
                        # The game will now show the final screen but won't accept more moves
                        # until reset.
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()