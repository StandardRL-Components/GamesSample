
# Generated: 2025-08-28T02:07:57.513290
# Source Brief: brief_01607.md
# Brief Index: 1607

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Space to attack the nearest monster."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a grid, strategically battling 7 monsters to clear the arena before your health runs out."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.UI_HEIGHT = 40
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 9
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000

        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_DAMAGE = 20
        self.MONSTER_COUNT = 7
        self.MONSTER_MAX_HEALTH = 30
        self.MONSTER_DAMAGE = 10

        # --- Colors ---
        self.COLOR_BG = (18, 22, 33)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_PLAYER = (66, 194, 255)
        self.COLOR_MONSTER = (255, 60, 60)
        self.COLOR_ATTACK = (255, 255, 100)
        self.COLOR_HIT = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BG = (70, 30, 30)
        self.COLOR_HEALTH_FG = (30, 200, 30)
        self.COLOR_MONSTER_HEALTH_FG = (200, 30, 30)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_game_over = pygame.font.SysFont('monospace', 48, bold=True)

        # --- Game State ---
        self.player_pos = [0, 0]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.monsters = []
        self.vfx = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        self.player_is_hit = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        self.vfx.clear()

        # Player setup
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_is_hit = False

        # Monster setup
        self.monsters.clear()
        possible_spawns = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_spawns.remove(tuple(self.player_pos))
        
        spawn_indices = self.np_random.choice(len(possible_spawns), self.MONSTER_COUNT, replace=False)
        spawn_points = [possible_spawns[i] for i in spawn_indices]

        for i, pos in enumerate(spawn_points):
            self.monsters.append({
                "pos": list(pos),
                "health": self.MONSTER_MAX_HEALTH,
                "id": i,
                "is_hit": False,
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Cost of living

        # Clear one-frame effects
        self.vfx.clear()
        self.player_is_hit = False
        for m in self.monsters:
            m["is_hit"] = False

        # --- 1. Player Action ---
        movement, space_held, _ = action
        attacked = space_held == 1
        
        if attacked:
            reward += self._handle_player_attack()
        elif movement > 0:
            self._handle_player_movement(movement)

        # --- 2. Update Monster State (remove dead) ---
        monsters_killed_this_turn = [m for m in self.monsters if m["health"] <= 0]
        if monsters_killed_this_turn:
            # sfx: monster_die.wav
            reward += 10 * len(monsters_killed_this_turn)
            self.monsters = [m for m in self.monsters if m["health"] > 0]

        # --- 3. Monster Actions ---
        if not self.game_over and len(self.monsters) > 0:
            reward += self._handle_monster_actions()

        # --- 4. Termination Check ---
        terminated = False
        if self.player_health <= 0:
            # sfx: player_death.wav
            reward += -100
            terminated = True
            self.game_over = True
            self.game_over_message = "YOU DIED"
        elif not self.monsters:
            # sfx: victory.wav
            reward += 100
            terminated = True
            self.game_over = True
            self.game_over_message = "VICTORY!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME'S UP"
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_attack(self):
        reward = 0
        target_monster = self._get_nearest_monster()
        if not target_monster:
            return 0

        # sfx: player_attack.wav
        
        # Risky attack bonus
        adjacent_monsters = self._count_adjacent_monsters(self.player_pos)
        if adjacent_monsters > 1:
            reward += 2.0

        target_monster["health"] -= self.PLAYER_DAMAGE
        target_monster["is_hit"] = True

        # Add attack VFX
        start_pos = self._grid_to_pixel(self.player_pos)
        end_pos = self._grid_to_pixel(target_monster["pos"])
        self.vfx.append({"type": "line", "start": start_pos, "end": end_pos, "color": self.COLOR_ATTACK, "width": 3})
        
        return reward

    def _handle_player_movement(self, movement):
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right

        if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
            self.player_pos = [px, py]

    def _handle_monster_actions(self):
        reward = 0
        all_monster_positions = {tuple(m["pos"]) for m in self.monsters}

        for monster in self.monsters:
            # --- Movement AI ---
            m_pos = monster["pos"]
            p_pos = self.player_pos
            
            # Find best moves
            potential_moves = []
            if p_pos[0] > m_pos[0]: potential_moves.append((m_pos[0] + 1, m_pos[1]))
            elif p_pos[0] < m_pos[0]: potential_moves.append((m_pos[0] - 1, m_pos[1]))
            if p_pos[1] > m_pos[1]: potential_moves.append((m_pos[0], m_pos[1] + 1))
            elif p_pos[1] < m_pos[1]: potential_moves.append((m_pos[0], m_pos[1] - 1))
            
            # Filter out moves into other monsters
            valid_moves = [move for move in potential_moves if move not in all_monster_positions]

            if valid_moves:
                # Choose randomly from the best moves
                move_idx = self.np_random.integers(len(valid_moves))
                new_pos = list(valid_moves[move_idx])
                
                # Update positions set for collision checking in the same frame
                all_monster_positions.remove(tuple(m_pos))
                all_monster_positions.add(tuple(new_pos))
                monster["pos"] = new_pos
            
            # --- Attack ---
            if self._are_adjacent(monster["pos"], self.player_pos):
                # sfx: player_hit.wav
                self.player_health -= self.MONSTER_DAMAGE
                self.player_is_hit = True
                reward -= 5.0
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "monsters_left": len(self.monsters),
        }

    def _render_game(self):
        grid_offset_y = self.UI_HEIGHT
        
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (x * self.CELL_SIZE, grid_offset_y)
            end = (x * self.CELL_SIZE, self.HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (0, y * self.CELL_SIZE + grid_offset_y)
            end = (self.WIDTH, y * self.CELL_SIZE + grid_offset_y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
            
        # Draw monsters
        for m in self.monsters:
            color = self.COLOR_HIT if m["is_hit"] else self.COLOR_MONSTER
            center = self._grid_to_pixel(m["pos"])
            radius = self.CELL_SIZE // 2 - 4
            
            # Shadow
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1] + 2, radius, (0,0,0,100))
            # Body
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
            
            # Monster health bar
            bar_w = self.CELL_SIZE * 0.8
            bar_h = 5
            bar_x = center[0] - bar_w / 2
            bar_y = center[1] - radius - 10
            health_pct = m["health"] / self.MONSTER_MAX_HEALTH
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_MONSTER_HEALTH_FG, (bar_x, bar_y, bar_w * health_pct, bar_h))
            
        # Draw player
        player_color = self.COLOR_HIT if self.player_is_hit else self.COLOR_PLAYER
        center = self._grid_to_pixel(self.player_pos)
        size = self.CELL_SIZE - 8
        rect = pygame.Rect(center[0] - size // 2, center[1] - size // 2, size, size)
        
        # Shadow
        shadow_rect = rect.copy()
        shadow_rect.move_ip(0, 2)
        pygame.draw.rect(self.screen, (0,0,0,100), shadow_rect, border_radius=4)
        # Body
        pygame.draw.rect(self.screen, player_color, rect, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), rect, width=2, border_radius=4)

        # Draw VFX
        for effect in self.vfx:
            if effect["type"] == "line":
                pygame.draw.line(self.screen, effect["color"], effect["start"], effect["end"], effect["width"])

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, self.WIDTH, self.UI_HEIGHT))
        
        # Player Health
        health_text = self.font_ui.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        bar_w, bar_h = 200, 20
        bar_x, bar_y = 100, (self.UI_HEIGHT - bar_h) // 2
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_w * health_pct, bar_h), border_radius=3)

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(right=self.WIDTH - 10, centery=self.UI_HEIGHT // 2)
        self.screen.blit(score_text, score_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_game_over.render(self.game_over_message, True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2 + self.UI_HEIGHT
        return int(x), int(y)

    def _get_nearest_monster(self):
        if not self.monsters:
            return None
        
        min_dist = float('inf')
        nearest_monsters = []
        
        for m in self.monsters:
            dist = self._manhattan_distance(self.player_pos, m["pos"])
            if dist < min_dist:
                min_dist = dist
                nearest_monsters = [m]
            elif dist == min_dist:
                nearest_monsters.append(m)
        
        return self.np_random.choice(nearest_monsters)

    def _count_adjacent_monsters(self, pos):
        count = 0
        for m in self.monsters:
            if self._are_adjacent(pos, m["pos"]):
                count += 1
        return count

    @staticmethod
    def _manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def _are_adjacent(pos1, pos2):
        return GameEnv._manhattan_distance(pos1, pos2) == 1

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Grid Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = [0, 0, 0] # No-op
    
    while not terminated:
        # --- Human Controls ---
        # Reset action at the start of each turn-based loop
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                
                # Since it's turn-based, we process one action at a time
                if movement > 0 or space > 0:
                    action = [movement, space, shift]
                    obs, reward, terminated, _, info = env.step(action)
                    print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {terminated}")
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if env.game_over:
            # Wait for a moment on the game over screen before closing
            pygame.time.wait(2000)
            terminated = True

    env.close()
    print("Game Over. Final Score:", info['score'])