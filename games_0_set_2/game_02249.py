
# Generated: 2025-08-28T04:13:18.735673
# Source Brief: brief_02249.md
# Brief Index: 2249

        
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
    """
    A fast-paced, real-time Sokoban-style puzzle game.

    The player must push all crates onto their designated target locations before the
    timer runs out. The game features smooth, interpolated graphics and particle
    effects for a polished arcade feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Push crates onto the green targets."
    )

    # Short, user-facing description of the game
    game_description = (
        "A real-time puzzle game. Push all the brown crates onto the green targets "
        "before the 60-second timer runs out!"
    )

    # Frames auto-advance at a fixed rate for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    CELL_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
    MAX_STEPS = 1800 # 60 seconds * 30 FPS

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_WALL = (60, 60, 75)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_CRATE = (139, 69, 19)
    COLOR_CRATE_BORDER = (101, 51, 14)
    COLOR_TARGET = (50, 205, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIMER_WARN = (255, 100, 100)
    
    # Rewards
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_CRATE_ON_TARGET = 5.0
    REWARD_DISTANCE_FACTOR = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)

        self.level_layout = [
            "WWWWWWWWWWWWWWWW",
            "W              W",
            "W P C   T      W",
            "W       W      W",
            "W C  W  T      W",
            "W    W         W",
            "W   CWW T      W",
            "W              W",
            "W              W",
            "WWWWWWWWWWWWWWWW",
        ]

        self._parse_level()
        self.reset()
        self.validate_implementation()

    def _parse_level(self):
        self.initial_player_pos = None
        self.initial_crates = []
        self.targets = []
        self.walls = []
        for y, row in enumerate(self.level_layout):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.walls.append(pos)
                elif char == 'P':
                    self.initial_player_pos = pos
                elif char == 'C':
                    self.initial_crates.append(pos)
                elif char == 'T':
                    self.targets.append(pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = self.initial_player_pos
        self.crates = list(self.initial_crates)
        
        # Visual positions for smooth interpolation
        self.visual_player_pos = [p * self.CELL_SIZE for p in self.player_pos]
        self.visual_crates_pos = [[p * self.CELL_SIZE for p in c] for c in self.crates]
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS / self.FPS
        
        self.particles = []
        self.newly_completed_targets = set()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0.0
        
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1 / self.FPS)

        # --- Game Logic ---
        if not self.game_over:
            # Store pre-move state for reward calculation
            old_crates_on_target_count = self._count_crates_on_targets()
            old_crate_distances = self._get_crate_distances_to_targets()

            # Handle player movement and crate pushing
            move_reward = self._handle_movement(movement, old_crate_distances)
            reward += move_reward

            # Check for newly completed targets
            new_crates_on_target_count = self._count_crates_on_targets()
            if new_crates_on_target_count > old_crates_on_target_count:
                # Find which crate just landed on a target
                for i, crate_pos in enumerate(self.crates):
                    if crate_pos in self.targets and crate_pos not in self.newly_completed_targets:
                        # SFX: Crate placed
                        reward += self.REWARD_CRATE_ON_TARGET
                        self._create_particles(crate_pos, self.COLOR_TARGET)
                        self.newly_completed_targets.add(crate_pos)
        
        self.score += reward
        
        # --- Update visual elements ---
        self._update_visuals()
        self._update_particles()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            is_win = self._count_crates_on_targets() == len(self.targets)
            if is_win:
                reward += self.REWARD_WIN
                self.score += self.REWARD_WIN
                # SFX: Level complete
            else: # Loss by time or steps
                reward += self.REWARD_LOSS
                self.score += self.REWARD_LOSS
                # SFX: Game over
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_movement(self, movement, old_crate_distances):
        if movement == 0: # No-op
            return 0.0

        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
        
        player_x, player_y = self.player_pos
        next_player_pos = (player_x + dx, player_y + dy)

        # Check for wall collision
        if next_player_pos in self.walls:
            # SFX: Bump wall
            return 0.0

        # Check for crate collision
        if next_player_pos in self.crates:
            crate_index = self.crates.index(next_player_pos)
            next_crate_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
            
            # Check if crate can be pushed
            if next_crate_pos not in self.walls and next_crate_pos not in self.crates:
                # SFX: Push crate
                old_crate_pos = self.crates[crate_index]
                self.crates[crate_index] = next_crate_pos
                self.player_pos = next_player_pos
                
                # Calculate distance-based reward
                new_crate_distances = self._get_crate_distances_to_targets()
                dist_before = old_crate_distances.get(old_crate_pos, float('inf'))
                dist_after = new_crate_distances.get(next_crate_pos, float('inf'))
                
                return (dist_before - dist_after) * self.REWARD_DISTANCE_FACTOR
            else:
                # SFX: Bump crate (can't push)
                return 0.0
        
        # Regular movement
        # SFX: Player step
        self.player_pos = next_player_pos
        return 0.0
    
    def _get_crate_distances_to_targets(self):
        distances = {}
        unoccupied_targets = [t for t in self.targets if t not in self.crates]
        if not unoccupied_targets:
            return {}
            
        for crate_pos in self.crates:
            if crate_pos in self.targets:
                distances[crate_pos] = 0
            else:
                min_dist = min(
                    abs(crate_pos[0] - t[0]) + abs(crate_pos[1] - t[1]) 
                    for t in unoccupied_targets
                )
                distances[crate_pos] = min_dist
        return distances

    def _check_termination(self):
        win = self._count_crates_on_targets() == len(self.targets)
        timeout = self.time_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        return win or timeout or max_steps_reached

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "crates_on_target": self._count_crates_on_targets(),
            "total_crates": len(self.crates),
        }

    def _count_crates_on_targets(self):
        return sum(1 for crate in self.crates if crate in self.targets)
    
    def _update_visuals(self):
        # Interpolate player
        target_px = self.player_pos[0] * self.CELL_SIZE
        target_py = self.player_pos[1] * self.CELL_SIZE
        self.visual_player_pos[0] += (target_px - self.visual_player_pos[0]) * 0.5
        self.visual_player_pos[1] += (target_py - self.visual_player_pos[1]) * 0.5

        # Interpolate crates
        for i, crate_pos in enumerate(self.crates):
            target_cx = crate_pos[0] * self.CELL_SIZE
            target_cy = crate_pos[1] * self.CELL_SIZE
            self.visual_crates_pos[i][0] += (target_cx - self.visual_crates_pos[i][0]) * 0.5
            self.visual_crates_pos[i][1] += (target_cy - self.visual_crates_pos[i][1]) * 0.5

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for x, y in self.targets:
            px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
            is_occupied = (x, y) in self.crates
            color = tuple(min(255, c + 50) for c in self.COLOR_TARGET) if is_occupied else self.COLOR_TARGET
            rect = pygame.Rect(px + 5, py + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-8, -8), border_radius=4)


        # Draw walls
        for x, y in self.walls:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw crates
        for i, (vx, vy) in enumerate(self.visual_crates_pos):
            rect = pygame.Rect(vx + 4, vy + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect, border_radius=6)
            pygame.draw.rect(self.screen, self.COLOR_CRATE_BORDER, rect, width=2, border_radius=6)

        # Draw player
        vx, vy = self.visual_player_pos
        player_rect = pygame.Rect(vx + 6, vy + 6, self.CELL_SIZE - 12, self.CELL_SIZE - 12)
        # Glow effect
        glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 30), (self.CELL_SIZE // 2, self.CELL_SIZE // 2), self.CELL_SIZE // 2 * 0.8)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 15), (self.CELL_SIZE // 2, self.CELL_SIZE // 2), self.CELL_SIZE // 2)
        self.screen.blit(glow_surf, (vx, vy))
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

        # Draw particles
        self._render_particles()

    def _render_ui(self):
        # Crates count
        crates_text = f"Crates: {self._count_crates_on_targets()}/{len(self.crates)}"
        text_surf = self.font_ui.render(crates_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        timer_str = f"{self.time_remaining:.1f}"
        timer_color = self.COLOR_TIMER_WARN if self.time_remaining < 10 else self.COLOR_TEXT
        timer_surf = self.font_timer.render(timer_str, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 8))
        self.screen.blit(timer_surf, timer_rect)

    def _create_particles(self, pos, color):
        px, py = pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2, pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 25)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            size = int(life_ratio * 5)
            if size > 0:
                alpha = int(life_ratio * 200)
                color = (*p['color'], alpha)
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (p['pos'][0] - size, p['pos'][1] - size), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        import os
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        
        pygame.display.set_caption(env.game_description)
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            movement = 0 # no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            # Action is a list: [movement, space, shift]
            action = [movement, 0, 0] 
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Wait a bit before closing
                pygame.time.wait(2000)

    except (ImportError, pygame.error) as e:
        print(f"Pygame display could not be initialized. Skipping manual play. Error: {e}")
        print("This is expected in a headless environment.")

    env.close()