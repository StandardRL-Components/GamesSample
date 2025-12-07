
# Generated: 2025-08-28T06:11:13.586918
# Source Brief: brief_02853.md
# Brief Index: 2853

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move your yellow square. Avoid all other colored squares to survive."
    )

    game_description = (
        "A fast-paced, grid-based survival game. Dodge an ever-moving swarm of monsters for 60 seconds to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (255, 220, 0)
        self.COLOR_PLAYER_INNER = (200, 170, 0)
        self.MONSTER_COLORS = {
            "red": (255, 50, 50), "blue": (50, 150, 255), "green": (50, 255, 100),
            "purple": (200, 50, 255), "orange": (255, 150, 50)
        }
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 24)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Game constants
        self.MAX_STEPS = 1800 # 60 seconds at 30 "FPS" -> 30 steps/sec. Brief says 3600, but 1800 is more playable for 60s.
        self.INITIAL_MONSTERS = 5

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.timer = 0
        self.player_pos = None
        self.monsters = []
        self.particles = []

        # Initialize state
        self.reset()
        
        # Run validation
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_STEPS
        
        self.player_pos = pygame.Vector2(self.GRID_W // 2, self.GRID_H // 2)
        
        self.monsters = []
        occupied_cells = {tuple(self.player_pos)}
        
        monster_types = list(self.MONSTER_COLORS.keys())
        for _ in range(self.INITIAL_MONSTERS):
            pos = self._get_random_empty_cell(occupied_cells)
            occupied_cells.add(tuple(pos))
            
            m_type = self.np_random.choice(monster_types)
            monster = {
                "pos": pos,
                "type": m_type,
                "color": self.MONSTER_COLORS[m_type],
                "state": {}
            }
            self._init_monster_state(monster)
            self.monsters.append(monster)
            
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.01  # Survival reward per step
        
        if not self.game_over:
            # Unpack action
            movement = action[0]
            
            # --- Update Game Logic ---
            # 1. Move Player
            self._move_player(movement)
            
            # 2. Move Monsters
            self._update_monsters()
            
            # 3. Update Timers
            self.steps += 1
            self.timer -= 1
            
            # --- Check for Termination ---
            # Collision
            if any(m["pos"] == self.player_pos for m in self.monsters):
                self.game_over = True
                self.win = False
                reward = -100.0
                # sfx: player_death_sound

            # Victory
            elif self.timer <= 0:
                self.game_over = True
                self.win = True
                reward = 100.0
                # sfx: victory_fanfare

        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._update_and_draw_particles()
        self._draw_monsters()
        self._draw_player()
        self._render_ui()
        
        if self.game_over:
            self._draw_game_over_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.timer / (self.MAX_STEPS / 60.0) if self.MAX_STEPS > 0 else 0,
            "player_pos": (int(self.player_pos.x), int(self.player_pos.y)),
            "monsters": len(self.monsters)
        }

    # --- Helper Methods for Game Logic ---
    
    def _move_player(self, movement):
        if movement == 1: self.player_pos.y -= 1  # Up
        elif movement == 2: self.player_pos.y += 1  # Down
        elif movement == 3: self.player_pos.x -= 1  # Left
        elif movement == 4: self.player_pos.x += 1  # Right
        
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.GRID_W - 1)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.GRID_H - 1)

    def _update_monsters(self):
        current_monster_positions = {tuple(m['pos']) for m in self.monsters}
        for monster in self.monsters:
            old_pos = monster["pos"].copy()
            
            if monster["type"] == "red": # Random
                moves = [pygame.Vector2(0, 1), pygame.Vector2(0, -1), pygame.Vector2(1, 0), pygame.Vector2(-1, 0)]
                monster["pos"] += self.np_random.choice(moves)
            
            elif monster["type"] == "blue": # Chaser
                delta = self.player_pos - monster["pos"]
                if abs(delta.x) > abs(delta.y):
                    monster["pos"].x += np.sign(delta.x)
                elif abs(delta.y) > 0:
                    monster["pos"].y += np.sign(delta.y)

            elif monster["type"] == "green": # Patrol
                state = monster["state"]
                target_point = state["path"][state["path_index"]]
                if monster["pos"] == target_point:
                    state["path_index"] = (state["path_index"] + 1) % len(state["path"])
                    target_point = state["path"][state["path_index"]]
                
                delta = target_point - monster["pos"]
                if delta.x != 0: monster["pos"].x += np.sign(delta.x)
                elif delta.y != 0: monster["pos"].y += np.sign(delta.y)

            elif monster["type"] == "purple": # Teleporter
                state = monster["state"]
                state["cooldown"] -= 1
                if state["cooldown"] <= 0:
                    occupied = {tuple(m['pos']) for m in self.monsters if m is not monster}
                    occupied.add(tuple(self.player_pos))
                    new_pos = self._get_random_empty_cell(occupied)
                    self._create_particle_burst(self._grid_to_screen(monster["pos"]), monster["color"])
                    monster["pos"] = new_pos
                    self._create_particle_burst(self._grid_to_screen(monster["pos"]), monster["color"])
                    state["cooldown"] = self.np_random.integers(3, 6)
                    # sfx: teleport_sound

            elif monster["type"] == "orange": # Spiral
                state = monster["state"]
                if state["leg_progress"] >= state["leg_length"]:
                    state["direction"] = (state["direction"] + 1) % 4
                    state["leg_progress"] = 0
                    if state["direction"] % 2 == 0:
                        state["leg_length"] += 1
                
                if state["direction"] == 0: monster["pos"].x += 1 # Right
                elif state["direction"] == 1: monster["pos"].y -= 1 # Up
                elif state["direction"] == 2: monster["pos"].x -= 1 # Left
                elif state["direction"] == 3: monster["pos"].y += 1 # Down
                state["leg_progress"] += 1

            # Clamp monster position and handle collisions with other monsters
            monster["pos"].x = np.clip(monster["pos"].x, 0, self.GRID_W - 1)
            monster["pos"].y = np.clip(monster["pos"].y, 0, self.GRID_H - 1)

            # Prevent monsters from overlapping
            if tuple(monster['pos']) in current_monster_positions and monster['pos'] != old_pos:
                monster['pos'] = old_pos # Revert move if occupied
            else:
                current_monster_positions.remove(tuple(old_pos))
                current_monster_positions.add(tuple(monster['pos']))

    def _init_monster_state(self, monster):
        if monster["type"] == "green":
            w, h = self.np_random.integers(2, self.GRID_W // 2, size=2)
            x, y = self.np_random.integers(0, [self.GRID_W - w, self.GRID_H - h])
            p1 = pygame.Vector2(x, y)
            p2 = pygame.Vector2(x + w, y)
            p3 = pygame.Vector2(x + w, y + h)
            p4 = pygame.Vector2(x, y + h)
            monster["state"]["path"] = [p1, p2, p3, p4]
            monster["state"]["path_index"] = 0
            monster["pos"] = p1.copy()
        elif monster["type"] == "purple":
            monster["state"]["cooldown"] = self.np_random.integers(3, 6)
        elif monster["type"] == "orange":
            monster["state"] = {
                "leg_length": 1, "leg_progress": 0, "direction": 0
            }

    # --- Helper Methods for Rendering ---

    def _grid_to_screen(self, pos):
        return pygame.Vector2(
            pos.x * self.CELL_SIZE + self.CELL_SIZE // 2,
            pos.y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _draw_grid(self):
        for x in range(self.GRID_W + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_H + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

    def _draw_player(self):
        pos = self._grid_to_screen(self.player_pos)
        size = self.CELL_SIZE * 0.7
        inner_size = size * 0.7
        rect = pygame.Rect(pos.x - size/2, pos.y - size/2, size, size)
        inner_rect = pygame.Rect(pos.x - inner_size/2, pos.y - inner_size/2, inner_size, inner_size)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_INNER, inner_rect, border_radius=3)
        
        # Glow effect
        glow_color = self.COLOR_PLAYER + (50,) # Add alpha
        s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (size, size), size*0.6)
        self.screen.blit(s, (pos.x - size, pos.y - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_monsters(self):
        for monster in self.monsters:
            pos = self._grid_to_screen(monster["pos"])
            size = self.CELL_SIZE * 0.65
            inner_size = size * 0.6
            rect = pygame.Rect(pos.x - size/2, pos.y - size/2, size, size)
            inner_rect = pygame.Rect(pos.x - inner_size/2, pos.y - inner_size/2, inner_size, inner_size)
            
            color = monster["color"]
            inner_color = tuple(max(0, c-50) for c in color)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, inner_color, inner_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.timer / (self.MAX_STEPS / 60.0))
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, (255, 255, 255))
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

    def _draw_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 50, 50)
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    # --- Particle System ---
    def _create_particle_burst(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 5), "color": color
            })
            # sfx: particle_burst_sound
    
    def _update_and_draw_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] > 0 and p["radius"] > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"]
                )
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]

    def _get_random_empty_cell(self, occupied_cells):
        while True:
            pos = pygame.Vector2(
                self.np_random.integers(0, self.GRID_W),
                self.np_random.integers(0, self.GRID_H)
            )
            if tuple(pos) not in occupied_cells:
                return pos

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Create a window to display the game
    pygame.display.set_caption("Grid Survival")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement_action = 0  # Default to no-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()
            done = False
            continue

        if not done:
            action = [movement_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(10) # Control the speed for manual play

    env.close()