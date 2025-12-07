import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:30:52.454426
# Source Brief: brief_01042.md
# Brief Index: 1042
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent navigates a shrinking crystal cave.
    The agent must solve crossword puzzles to unlock new paths, grow crystals
    to traverse gaps, and descend to deeper levels before being consumed by
    the collapsing cave.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore a shrinking crystal cave, solve crossword puzzles to unlock paths, and grow crystals to traverse gaps "
        "while descending to deeper levels."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to grow a crystal or interact with a puzzle. "
        "Use shift to cycle through letters when solving a puzzle."
    )
    auto_advance = True

    # --- Constants ---
    # Sizes
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 60
    WORLD_HEIGHT = 40
    TILE_SIZE = 40
    
    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_WALL = (60, 40, 80)
    COLOR_FLOOR = (30, 20, 50)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    COLOR_CRYSTAL = (255, 255, 255)
    COLOR_RESOURCE = (255, 223, 0)
    COLOR_EXIT = (148, 0, 211)
    COLOR_PUZZLE_CONSOLE = (0, 255, 127)
    COLOR_SHRINK_WALL = (255, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (0, 0, 0, 150)
    COLOR_PUZZLE_SOLVED = (0, 255, 0)
    COLOR_PUZZLE_ACTIVE = (255, 165, 0)

    # Game parameters
    MAX_STEPS = 2500
    TARGET_DEPTH = 5
    PLAYER_SPEED = 0.2
    PARTICLE_LIFESPAN = 30

    # Tile IDs
    T_FLOOR = 0
    T_WALL = 1
    T_CRYSTAL = 2
    T_RESOURCE = 3
    T_EXIT = 4
    T_CONSOLE = 5

    PUZZLE_WORDS = [
        "GYM", "CAVE", "CODE", "AGENT", "SOLVE", "DEEP", 
        "SHINE", "GLOW", "LEARN", "REWARD", "POLICY"
    ]

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables are initialized in reset()
        self.world = None
        self.player_pos = None
        self.player_visual_pos = None
        self.resources = None
        self.depth = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.cave_center = None
        self.initial_cave_radius = None
        self.cave_radius = None
        self.particles = None
        self.puzzle_state = None
        self.last_space_held = False
        self.last_shift_held = False
        self.player_state = "EXPLORING" # or "SOLVING_PUZZLE"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.depth = 1
        self.resources = 5
        self.game_over = False
        self.particles = deque()
        self.player_state = "EXPLORING"
        self.last_space_held = False
        self.last_shift_held = False

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # 1. Create a solid wall world
        self.world = np.full((self.WORLD_WIDTH, self.WORLD_HEIGHT), self.T_WALL, dtype=np.uint8)

        # 2. Carve out a cavern using random walks
        px, py = self.np_random.integers(1, self.WORLD_WIDTH - 1), self.np_random.integers(1, self.WORLD_HEIGHT - 1)
        self.player_pos = np.array([px, py], dtype=int)
        self.player_visual_pos = self.player_pos.astype(float)
        self.world[px, py] = self.T_FLOOR
        
        num_walks = 2000
        for _ in range(num_walks):
            dx, dy = self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1])
            if self.np_random.random() < 0.5:
                px = np.clip(px + dx, 1, self.WORLD_WIDTH - 2)
            else:
                py = np.clip(py + dy, 1, self.WORLD_HEIGHT - 2)
            self.world[px, py] = self.T_FLOOR
        
        self.cave_center = self.player_pos.copy()
        self.initial_cave_radius = max(self.WORLD_WIDTH, self.WORLD_HEIGHT) * self.TILE_SIZE * 0.7
        self.cave_radius = self.initial_cave_radius

        # 3. Place exit, console, and resources
        floor_tiles = [tuple(t) for t in np.argwhere(self.world == self.T_FLOOR)]
        self.np_random.shuffle(floor_tiles)
        
        # Place Exit far from player
        exit_pos = max(floor_tiles, key=lambda pos: np.linalg.norm(np.array(pos) - self.player_pos))
        self.world[exit_pos[0], exit_pos[1]] = self.T_EXIT

        # Place Puzzle Console near exit
        console_candidates = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = exit_pos[0] + dx, exit_pos[1] + dy
            if self.world[nx, ny] == self.T_FLOOR:
                console_candidates.append((nx, ny))
        if console_candidates:
            console_pos_idx = self.np_random.choice(len(console_candidates))
            console_pos = console_candidates[console_pos_idx]
            self.world[console_pos[0], console_pos[1]] = self.T_CONSOLE
            self.puzzle_console_pos = console_pos
        else: # Failsafe
            self.world[exit_pos[0]-1, exit_pos[1]] = self.T_CONSOLE
            self.puzzle_console_pos = (exit_pos[0]-1, exit_pos[1])

        # Place Resources
        resource_count = 10 - self.depth
        floor_tiles_set = set(floor_tiles) - {tuple(self.player_pos), exit_pos, self.puzzle_console_pos}
        
        for _ in range(resource_count):
            if not floor_tiles_set: break
            pos = floor_tiles_set.pop()
            self.world[pos[0], pos[1]] = self.T_RESOURCE
        
        # 4. Setup puzzle for the level
        clue_len = 3 + (self.depth -1) // 2
        word_candidates = [w for w in self.PUZZLE_WORDS if len(w) == clue_len]
        if not word_candidates: word_candidates = [self.PUZZLE_WORDS[-1]] # Failsafe
        
        self.puzzle_state = {
            "word": self.np_random.choice(word_candidates),
            "guess": ['_'] * clue_len,
            "active_slot": 0,
            "current_letter": 'A',
            "solved": False
        }

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        on_console = tuple(self.player_pos) == self.puzzle_console_pos
        if on_console and not self.puzzle_state["solved"]:
            self.player_state = "SOLVING_PUZZLE"
        else:
            self.player_state = "EXPLORING"

        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        if self.player_state == "SOLVING_PUZZLE":
            if shift_press:
                # Sfx: UI_Cycle_Letter
                current_ord = ord(self.puzzle_state["current_letter"])
                self.puzzle_state["current_letter"] = chr((current_ord - 65 + 1) % 26 + 65)
            if space_press:
                slot = self.puzzle_state["active_slot"]
                self.puzzle_state["guess"][slot] = self.puzzle_state["current_letter"]
                
                if self.puzzle_state["active_slot"] < len(self.puzzle_state["word"]) - 1:
                    self.puzzle_state["active_slot"] += 1
                    self.puzzle_state["current_letter"] = 'A'
                    # Sfx: UI_Confirm_Letter
                else: 
                    if "".join(self.puzzle_state["guess"]) == self.puzzle_state["word"]:
                        self.puzzle_state["solved"] = True
                        reward += 1.0
                        self.score += 100
                        # Sfx: Puzzle_Solved
                        for _ in range(50): self._create_particle(self.player_visual_pos * self.TILE_SIZE, self.COLOR_PUZZLE_SOLVED, 5)
                    else:
                        self.puzzle_state["guess"] = ['_'] * len(self.puzzle_state["word"])
                        self.puzzle_state["active_slot"] = 0
                        self.puzzle_state["current_letter"] = 'A'
                        # Sfx: Puzzle_Failed
                        for _ in range(30): self._create_particle(self.player_visual_pos * self.TILE_SIZE, self.COLOR_SHRINK_WALL, 3)

        elif self.player_state == "EXPLORING":
            target_pos = self.player_pos.copy()
            if movement == 1: target_pos[1] -= 1
            elif movement == 2: target_pos[1] += 1
            elif movement == 3: target_pos[0] -= 1
            elif movement == 4: target_pos[0] += 1

            if self.world[target_pos[0], target_pos[1]] != self.T_WALL:
                if self.world[target_pos[0], target_pos[1]] == self.T_EXIT and not self.puzzle_state["solved"]:
                    pass
                else:
                    self.player_pos = target_pos

            if space_press and self.resources > 0:
                if self.world[self.player_pos[0], self.player_pos[1]] == self.T_FLOOR:
                    self.world[self.player_pos[0], self.player_pos[1]] = self.T_CRYSTAL
                    self.resources -= 1
                    # Sfx: Crystal_Grow
                    for _ in range(30): self._create_particle(self.player_visual_pos * self.TILE_SIZE, self.COLOR_CRYSTAL, 4)


        self.steps += 1
        
        self.player_visual_pos += (self.player_pos - self.player_visual_pos) * self.PLAYER_SPEED

        for p in list(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        dist_before = np.linalg.norm((self.player_visual_pos - self.cave_center) * self.TILE_SIZE)
        shrink_rate = self.initial_cave_radius / (self.MAX_STEPS * 1.5)
        self.cave_radius -= shrink_rate
        dist_after = np.linalg.norm((self.player_visual_pos - self.cave_center) * self.TILE_SIZE)
        
        if dist_after > dist_before and dist_after > self.cave_radius - 100:
             reward -= 0.1

        current_tile_type = self.world[self.player_pos[0], self.player_pos[1]]
        if current_tile_type == self.T_RESOURCE:
            self.world[self.player_pos[0], self.player_pos[1]] = self.T_FLOOR
            self.resources += 1
            self.score += 10
            reward += 0.1
            # Sfx: Resource_Collect
            for _ in range(20): self._create_particle(self.player_visual_pos * self.TILE_SIZE, self.COLOR_RESOURCE, 3)

        elif current_tile_type == self.T_EXIT and self.puzzle_state["solved"]:
            reward += 5.0
            self.score += 500
            if self.depth >= self.TARGET_DEPTH:
                terminated = True
                self.game_over = True
                reward += 100.0
            else:
                self.depth += 1
                # Sfx: Level_Complete
                self._generate_level()
                self.player_state = "EXPLORING"

        player_dist_from_center = np.linalg.norm((self.player_visual_pos - self.cave_center) * self.TILE_SIZE)
        if player_dist_from_center > self.cave_radius:
            terminated = True
            self.game_over = True
            reward -= 100.0
            # Sfx: Player_Death

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _create_particle(self, pos, color, speed_mult):
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * speed_mult + 1
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': self.PARTICLE_LIFESPAN, 'color': color})

    def _get_observation(self):
        cam_x = self.player_visual_pos[0] * self.TILE_SIZE - self.SCREEN_WIDTH / 2
        cam_y = self.player_visual_pos[1] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2

        self.screen.fill(self.COLOR_BG)
        
        start_x = max(0, int(cam_x / self.TILE_SIZE))
        end_x = min(self.WORLD_WIDTH, int((cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 1)
        start_y = max(0, int(cam_y / self.TILE_SIZE))
        end_y = min(self.WORLD_HEIGHT, int((cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 1)

        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                tile_type = self.world[x, y]
                screen_pos = (int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y))
                rect = pygame.Rect(screen_pos, (self.TILE_SIZE, self.TILE_SIZE))
                
                color = self.COLOR_FLOOR
                if tile_type == self.T_WALL:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif tile_type == self.T_CRYSTAL:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    points = [
                        (rect.centerx, rect.top), (rect.right, rect.centery),
                        (rect.centerx, rect.bottom), (rect.left, rect.centery)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_CRYSTAL, points)
                elif tile_type == self.T_RESOURCE:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    pulse = abs(math.sin(self.steps * 0.1))
                    pygame.draw.circle(self.screen, self.COLOR_RESOURCE, rect.center, int(self.TILE_SIZE * 0.3 + pulse * 3))
                elif tile_type == self.T_EXIT:
                    color = self.COLOR_EXIT if not self.puzzle_state["solved"] else self.COLOR_PUZZLE_SOLVED
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    pygame.draw.circle(self.screen, color, rect.center, int(self.TILE_SIZE * 0.4), 4)
                elif tile_type == self.T_CONSOLE:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                    color = self.COLOR_PUZZLE_CONSOLE if not self.puzzle_state["solved"] else self.COLOR_PUZZLE_SOLVED
                    pygame.draw.rect(self.screen, color, rect.inflate(-8, -8))
                else:
                    pygame.draw.rect(self.screen, color, rect)

        player_screen_pos = (
            int(self.player_visual_pos[0] * self.TILE_SIZE - cam_x),
            int(self.player_visual_pos[1] * self.TILE_SIZE - cam_y)
        )
        glow_radius = int(self.TILE_SIZE * 0.6)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_screen_pos[0] - glow_radius, player_screen_pos[1] - glow_radius))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_screen_pos, int(self.TILE_SIZE * 0.4))

        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
            alpha = max(0, int(255 * (p['life'] / self.PARTICLE_LIFESPAN)))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color_with_alpha)

        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 255))
        cave_center_screen = (
            int(self.cave_center[0] * self.TILE_SIZE - cam_x),
            int(self.cave_center[1] * self.TILE_SIZE - cam_y)
        )
        pygame.draw.circle(overlay, (0, 0, 0, 0), cave_center_screen, int(self.cave_radius))
        self.screen.blit(overlay, (0, 0))
        if self.cave_radius < self.initial_cave_radius * 0.98:
            pygame.gfxdraw.aacircle(self.screen, cave_center_screen[0], cave_center_screen[1], int(self.cave_radius), self.COLOR_SHRINK_WALL)
            pygame.gfxdraw.aacircle(self.screen, cave_center_screen[0], cave_center_screen[1], int(self.cave_radius)-1, self.COLOR_SHRINK_WALL)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        ui_bg_rect = pygame.Rect(5, 5, 220, 85)
        ui_bg_surf = pygame.Surface(ui_bg_rect.size, pygame.SRCALPHA)
        ui_bg_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bg_surf, ui_bg_rect.topleft)
        
        depth_text = self.font_medium.render(f"DEPTH: {self.depth}/{self.TARGET_DEPTH}", True, self.COLOR_UI_TEXT)
        resource_text = self.font_medium.render(f"CRYSTALS: {self.resources}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(depth_text, (15, 15))
        self.screen.blit(resource_text, (15, 40))
        self.screen.blit(score_text, (15, 65))

        puzzle_guess = self.puzzle_state["guess"]
        puzzle_len = len(puzzle_guess)
        
        puzzle_bg_rect = pygame.Rect(self.SCREEN_WIDTH - 20 - (puzzle_len * 25), 5, puzzle_len * 25 + 15, 65)
        puzzle_bg_surf = pygame.Surface(puzzle_bg_rect.size, pygame.SRCALPHA)
        puzzle_bg_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(puzzle_bg_surf, puzzle_bg_rect.topleft)

        title_text = self.font_small.render("PUZZLE", True, self.COLOR_UI_TEXT)
        self.screen.blit(title_text, (puzzle_bg_rect.left + 10, puzzle_bg_rect.top + 5))

        for i, char in enumerate(puzzle_guess):
            letter_text = self.font_medium.render(char, True, self.COLOR_UI_TEXT)
            x_pos = puzzle_bg_rect.left + 10 + i * 25
            y_pos = puzzle_bg_rect.top + 30
            self.screen.blit(letter_text, (x_pos, y_pos))
            if self.player_state == "SOLVING_PUZZLE" and i == self.puzzle_state["active_slot"]:
                pygame.draw.rect(self.screen, self.COLOR_PUZZLE_ACTIVE, (x_pos-2, y_pos-2, 22, 28), 2)
                sel_letter_text = self.font_medium.render(self.puzzle_state["current_letter"], True, self.COLOR_PUZZLE_ACTIVE)
                self.screen.blit(sel_letter_text, (x_pos, y_pos))

        if self.game_over:
            outcome_text = "YOU WIN!" if self.depth >= self.TARGET_DEPTH else "GAME OVER"
            rendered_text = self.font_large.render(outcome_text, True, self.COLOR_UI_TEXT)
            text_rect = rendered_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(rendered_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "depth": self.depth,
            "resources": self.resources,
            "puzzle_solved": self.puzzle_state["solved"],
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # The main loop is for human play and visualization, not for training.
    # It will not run in a headless environment.
    # To test the environment, you would typically use a script that
    # creates the environment and runs steps, like:
    #
    # import gymnasium as gym
    # env = GameEnv()
    # obs, info = env.reset()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
    # env.close()
    
    # However, to allow for interactive play, we can check for a display.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        print("Running in headless mode. Skipping interactive play.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        done = False
        
        pygame.display.set_caption("Crystal Cave Crossword")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        action = [0, 0, 0]
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            
            action[0] = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
                
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30)

        env.close()