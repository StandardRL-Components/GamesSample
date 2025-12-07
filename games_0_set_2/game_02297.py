
# Generated: 2025-08-28T04:23:08.331017
# Source Brief: brief_02297.md
# Brief Index: 2297

        
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
        "On your turn, you can either move (arrow keys) or attack (space + arrow key). Monsters move after your turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a tactical grid-based arena. Plan your moves and attacks carefully to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        
        self.MAX_STEPS = 1000
        self.MAX_WAVES = 5
        self.PLAYER_MAX_HEALTH = 10
        self.MONSTER_MAX_HEALTH = 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_MONSTER = (200, 50, 50)
        self.COLOR_ATTACK = (250, 250, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HIT = (255, 255, 255)
        self.COLOR_HEALTH_BAR_BG = (70, 20, 20)
        self.COLOR_HEALTH_BAR_FILL = (200, 40, 40)
        self.COLOR_PLAYER_HEALTH_FILL = (40, 200, 40)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Direction mapping
        self.DIRECTIONS = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
        }

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_facing = None
        self.monsters = None
        self.wave = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.hit_flashes = None
        self.screen_shake = None
        
        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing = (0, -1) # Start facing up
        self.monsters = []
        self.wave = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.hit_flashes = []
        self.screen_shake = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty per step to encourage efficiency
        self.steps += 1

        # --- Player Turn ---
        movement, space_held, _ = action
        is_attack_action = space_held == 1

        if is_attack_action:
            attack_dir_key = movement
            reward += self._handle_player_attack(attack_dir_key)
        else:
            move_dir_key = movement
            reward += self._handle_player_move(move_dir_key)

        # --- Monster Turn ---
        if not self.game_over:
            reward += self._handle_monster_turns()
        
        # --- State Update ---
        reward += self._update_game_state()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Died due to steps
             reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_attack(self, direction_key):
        reward = 0
        attack_dir = self.DIRECTIONS.get(direction_key, self.player_facing)
        self.player_facing = attack_dir
        
        target_pos = [self.player_pos[0] + attack_dir[0], self.player_pos[1] + attack_dir[1]]
        
        # Create attack visual effect
        # sound: player_attack.wav
        start_px = self._grid_to_pixels(self.player_pos)
        end_px = self._grid_to_pixels(target_pos)
        self._create_beam_particles(start_px, end_px, self.COLOR_ATTACK)

        for monster in self.monsters:
            if monster["pos"] == target_pos:
                monster["health"] -= 1
                reward += 10
                self.hit_flashes.append({"entity": monster, "duration": 5})
                # sound: monster_hit.wav
                if monster["health"] <= 0:
                    self.score += 50
        return reward

    def _handle_player_move(self, direction_key):
        reward = 0
        if direction_key == 0: # No-op
            return reward

        move_dir = self.DIRECTIONS[direction_key]
        self.player_facing = move_dir
        
        # Get distance to nearest monster before move
        dist_before = self._get_min_monster_dist()

        new_pos = [self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1]]

        if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
            self.player_pos = new_pos
            # Get distance after move
            dist_after = self._get_min_monster_dist()
            if dist_after < dist_before:
                reward += 0.1 # Reward for getting closer
        
        return reward

    def _handle_monster_turns(self):
        reward = 0
        for monster in self.monsters:
            dx = self.player_pos[0] - monster["pos"][0]
            dy = self.player_pos[1] - monster["pos"][1]
            
            if abs(dx) + abs(dy) == 1: # Adjacent, so attack
                self.player_health -= 1
                reward -= 5
                self.hit_flashes.append({"entity": "player", "duration": 5})
                self.screen_shake = 5
                # sound: player_hit.wav
                if self.player_health <= 0:
                    self.game_over = True
                    reward -= 100
                    # sound: game_over.wav
            else: # Not adjacent, so move
                if abs(dx) > abs(dy):
                    monster["pos"][0] += np.sign(dx)
                elif abs(dy) > 0:
                    monster["pos"][1] += np.sign(dy)
        return reward

    def _update_game_state(self):
        reward = 0
        # Remove dead monsters
        initial_monster_count = len(self.monsters)
        self.monsters = [m for m in self.monsters if m["health"] > 0]
        defeated_count = initial_monster_count - len(self.monsters)
        if defeated_count > 0:
            self.score += defeated_count * 25 # Bonus for kill
            # sound: monster_die.wav

        # Check for wave clear
        if not self.monsters and not self.game_over:
            self.score += 100
            reward += 100
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                self.game_over = True
                # sound: victory.wav
            else:
                self._spawn_wave()
                # sound: wave_clear.wav
        return reward

    def _get_observation(self):
        # Apply screen shake
        render_offset = [0, 0]
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset[0] = self.np_random.integers(-5, 6)
            render_offset[1] = self.np_random.integers(-5, 6)

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Create a separate surface for the game world to apply shake
        game_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        
        # Render all game elements
        self._render_game(game_surface)
        
        # Blit the game surface with the shake offset
        self.screen.blit(game_surface, render_offset)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, surface):
        self._draw_grid(surface)
        self._update_and_draw_particles(surface)
        self._draw_monsters(surface)
        self._draw_player(surface)
        self._update_hit_flashes()

    def _draw_grid(self, surface):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y)
            end_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y + self.GRID_HEIGHT)
            pygame.draw.line(surface, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_ORIGIN_X + self.GRID_WIDTH, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(surface, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_player(self, surface):
        player_px, player_py = self._grid_to_pixels(self.player_pos)
        color = self.COLOR_HIT if self._is_flashing("player") else self.COLOR_PLAYER
        
        player_rect = pygame.Rect(player_px - 15, player_py - 15, 30, 30)
        pygame.draw.rect(surface, color, player_rect, border_radius=4)
        
        # Facing indicator
        face_dx, face_dy = self.player_facing
        indicator_start = (player_px, player_py)
        indicator_end = (player_px + face_dx * 12, player_py + face_dy * 12)
        pygame.draw.line(surface, self.COLOR_BG, indicator_start, indicator_end, 4)

    def _draw_monsters(self, surface):
        for monster in self.monsters:
            monster_px, monster_py = self._grid_to_pixels(monster["pos"])
            color = self.COLOR_HIT if self._is_flashing(monster) else self.COLOR_MONSTER
            
            # Body
            points = [
                (monster_px, monster_py - 15),
                (monster_px + 15, monster_py + 10),
                (monster_px - 15, monster_py + 10),
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

            # Health bar
            if monster["health"] < self.MONSTER_MAX_HEALTH:
                bar_width = 24
                bar_height = 4
                health_pct = monster["health"] / self.MONSTER_MAX_HEALTH
                bg_rect = pygame.Rect(monster_px - bar_width // 2, monster_py + 15, bar_width, bar_height)
                fill_rect = pygame.Rect(monster_px - bar_width // 2, monster_py + 15, int(bar_width * health_pct), bar_height)
                pygame.draw.rect(surface, self.COLOR_HEALTH_BAR_BG, bg_rect)
                pygame.draw.rect(surface, self.COLOR_HEALTH_BAR_FILL, fill_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 10))

        # Wave
        wave_text = self.font_large.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 10))

        # Player Health
        health_text = self.font_small.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (20, 10))
        bar_width = 150
        bar_height = 20
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bg_rect = pygame.Rect(20, 35, bar_width, bar_height)
        fill_rect = pygame.Rect(20, 35, int(bar_width * health_pct), bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HEALTH_FILL, fill_rect, border_radius=3)
        
        if self.game_over:
            outcome_text = "VICTORY!" if self.wave > self.MAX_WAVES else "GAME OVER"
            text_surface = self.font_large.render(outcome_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "monsters_left": len(self.monsters),
        }

    def _spawn_wave(self):
        self.monsters.clear()
        num_monsters = min(self.GRID_SIZE, self.wave + 1)
        
        occupied_cells = [tuple(self.player_pos)]
        for _ in range(num_monsters):
            pos = None
            while pos is None or tuple(pos) in occupied_cells:
                pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE)]
            
            occupied_cells.append(tuple(pos))
            self.monsters.append({"pos": pos, "health": self.MONSTER_MAX_HEALTH})

    def _grid_to_pixels(self, grid_pos):
        px = self.GRID_ORIGIN_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_ORIGIN_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _get_min_monster_dist(self):
        if not self.monsters:
            return float('inf')
        return min(abs(self.player_pos[0] - m["pos"][0]) + abs(self.player_pos[1] - m["pos"][1]) for m in self.monsters)

    def _create_beam_particles(self, start_pos, end_pos, color):
        dist = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        steps = max(1, int(dist / 5))
        for i in range(steps + 1):
            t = i / steps
            px = start_pos[0] * (1 - t) + end_pos[0] * t
            py = start_pos[1] * (1 - t) + end_pos[1] * t
            self.particles.append({
                "pos": [px, py],
                "vel": [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)],
                "life": 10,
                "color": color,
                "radius": self.np_random.integers(2, 5)
            })

    def _update_and_draw_particles(self, surface):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / 10))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
                surface.blit(temp_surf, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]), special_flags=pygame.BLEND_RGBA_ADD)

    def _update_hit_flashes(self):
        for flash in self.hit_flashes[:]:
            flash["duration"] -= 1
            if flash["duration"] <= 0:
                self.hit_flashes.remove(flash)

    def _is_flashing(self, entity_id):
        return any(f["entity"] == entity_id for f in self.hit_flashes)

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Combat Arena")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                if terminated:
                    continue

                # Get keyboard state for simultaneous presses
                keys = pygame.key.get_pressed()
                
                # Determine action based on keys
                movement = 0
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                space = 1 if keys[pygame.K_SPACE] else 0
                shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                
                action = [movement, space, shift]
                
                # A single key press triggers one step in this turn-based game
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    env.close()