import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:44:34.787944
# Source Brief: brief_01924.md
# Brief Index: 1924
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide two characters through a maze in mirrored movement. "
        "Work together to activate switches, open gates, and reach the exit simultaneously."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Both characters move in mirrored directions. "
        "Cooperate to solve the maze."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 2700  # 90 seconds * 30 FPS
        self.MOVE_COOLDOWN_STEPS = 5 # Cooldown to make movement feel deliberate and controllable

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (40, 60, 85)
        self.COLOR_CHAR1 = (255, 60, 60)
        self.COLOR_CHAR2 = (255, 200, 60)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_SWITCH_OFF = (120, 120, 120)
        self.COLOR_SWITCH_ON = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # --- Maze Definition ---
        # 0: empty, 1: wall, 2: char1 start, 3: char2 start, 4: exit
        # 5: switch1, 6: gate1, 7: switch2, 8: gate2
        self.maze_layout = [
            "11111111111111111111111111111111",
            "12000000000001100000000000000001",
            "10111111111111101111111111111101",
            "10100000000000000000000000000101",
            "10101111111110111111111111101181",
            "10001000000000000000000000100001",
            "11111011111111111111111111101111",
            "15000010000000000000000001000001",
            "11111011111111111111111111101111",
            "10001000000000000000000000100001",
            "10101111111110111111111111101111",
            "10100000000000000000000000000101",
            "10111111111111101111111111111101",
            "10000000000001100000000000000001",
            "10111111111111111111111111111101",
            "10000000000007000000000000000001",
            "10111111111111111111111111111101",
            "10000000000000000000000000000041",
            "13000000000000000000000000000041",
            "11111111111111111111111111111111",
        ]
        
        self._parse_maze()

    def _parse_maze(self):
        self.walls, self.switches, self.gates = [], {}, {}
        switch_gate_map = {}
        
        for r, row_str in enumerate(self.maze_layout):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == '1': self.walls.append(pos)
                elif char == '2': self.char1_start_pos = np.array(pos)
                elif char == '3': self.char2_start_pos = np.array(pos)
                elif char == '4': self.exit_pos = np.array(pos)
                elif char.isdigit() and int(char) >= 5 and int(char) % 2 != 0:
                    switch_id = int(char)
                    self.switches[pos] = {'state': False, 'gates': []}
                    if switch_id not in switch_gate_map: switch_gate_map[switch_id] = []
                elif char.isdigit() and int(char) >= 6 and int(char) % 2 == 0:
                    gate_id, switch_id = int(char), int(char) - 1
                    self.gates[pos] = switch_id
                    if switch_id not in switch_gate_map: switch_gate_map[switch_id] = []
                    switch_gate_map[switch_id].append(pos)

        for pos, data in self.switches.items():
            switch_id = self._get_id_at_pos(pos)
            if switch_id in switch_gate_map: data['gates'] = switch_gate_map[switch_id]

    def _get_id_at_pos(self, pos):
        c, r = pos
        if 0 <= r < len(self.maze_layout) and 0 <= c < len(self.maze_layout[r]):
            char = self.maze_layout[r][c]
            if char.isdigit(): return int(char)
        return -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.char1_pos = self.char1_start_pos.copy()
        self.char2_pos = self.char2_start_pos.copy()
        
        self.char1_visual_pos = self.char1_pos.astype(float) * self.GRID_SIZE
        self.char2_visual_pos = self.char2_pos.astype(float) * self.GRID_SIZE
        
        self.time_remaining = 90.0
        self.sync_actions = 0
        self.total_moves = 0
        self.move_cooldown = 0
        
        for switch_pos in self.switches:
            self.switches[switch_pos]['state'] = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = False
        
        if self.move_cooldown > 0: self.move_cooldown -= 1
            
        movement = action[0]
        
        if self.move_cooldown == 0 and movement != 0:
            self.move_cooldown = self.MOVE_COOLDOWN_STEPS
            self.total_moves += 1

            move_map = {1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([-1, 0]), 4: np.array([1, 0])}
            char1_move = move_map.get(movement, np.array([0,0]))
            char2_move = -char1_move

            if movement != 0:
                self.sync_actions += 1
                reward += 0.1  # Synchronization reward

            next_pos1, next_pos2 = self.char1_pos + char1_move, self.char2_pos + char2_move
            
            is_valid1 = not self._is_collision(next_pos1)
            is_valid2 = not self._is_collision(next_pos2)
            
            if np.array_equal(next_pos1, self.char2_pos) and np.array_equal(next_pos2, self.char1_pos): is_valid1 = is_valid2 = False
            if np.array_equal(next_pos1, next_pos2): is_valid1 = is_valid2 = False
            
            if is_valid1: self.char1_pos = next_pos1
            if is_valid2: self.char2_pos = next_pos2
        
        newly_activated_switches = self._update_switches()
        if newly_activated_switches:
            reward += 1.0 * newly_activated_switches
            # SFX: Switch Activate

        self.steps += 1
        self.time_remaining = max(0, 90.0 - (self.steps / 30.0))
        self.score += reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if np.array_equal(self.char1_pos, self.exit_pos) and np.array_equal(self.char2_pos, self.exit_pos):
                time_bonus = 100 * (self.time_remaining / 90.0)
                self.score += time_bonus
                reward += time_bonus
                # SFX: Level Complete
            else:
                # SFX: Game Over
                pass

        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _is_collision(self, pos):
        if tuple(pos) in self.walls: return True
        if tuple(pos) in self.gates:
            switch_id = self.gates[tuple(pos)]
            for switch_pos, data in self.switches.items():
                if self._get_id_at_pos(switch_pos) == switch_id:
                    if not data['state']: return True
                    break
        if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT): return True
        return False
        
    def _update_switches(self):
        activated_count = 0
        for pos, data in self.switches.items():
            if not data['state']:
                adj_p1 = abs(self.char1_pos[0] - pos[0]) + abs(self.char1_pos[1] - pos[1]) == 1
                adj_p2 = abs(self.char2_pos[0] - pos[0]) + abs(self.char2_pos[1] - pos[1]) == 1
                if adj_p1 and adj_p2:
                    data['state'] = True
                    activated_count += 1
        return activated_count

    def _check_termination(self):
        win = np.array_equal(self.char1_pos, self.exit_pos) and np.array_equal(self.char2_pos, self.exit_pos)
        timeout = self.steps >= self.MAX_STEPS
        return win or timeout

    def _get_observation(self):
        lerp_factor = 0.4
        target_visual1 = self.char1_pos.astype(float) * self.GRID_SIZE
        target_visual2 = self.char2_pos.astype(float) * self.GRID_SIZE
        self.char1_visual_pos += (target_visual1 - self.char1_visual_pos) * lerp_factor
        self.char2_visual_pos += (target_visual2 - self.char2_visual_pos) * lerp_factor

        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                pos = (c, r)
                rect = pygame.Rect(c * self.GRID_SIZE, r * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                
                if pos in self.walls: pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif pos in self.gates:
                    is_on = False
                    switch_id = self.gates[pos]
                    for switch_pos, data in self.switches.items():
                        if self._get_id_at_pos(switch_pos) == switch_id: is_on = data['state']; break
                    if not is_on: pygame.draw.rect(self.screen, self.COLOR_WALL, rect, border_radius=4)
                    else: pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-self.GRID_SIZE*0.7, -self.GRID_SIZE*0.7))
                elif pos in self.switches:
                    is_on = self.switches[pos]['state']
                    color = self.COLOR_SWITCH_ON if is_on else self.COLOR_SWITCH_OFF
                    center_x, center_y = int(rect.centerx), int(rect.centery)
                    radius = self.GRID_SIZE // 3
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    if is_on:
                        for i in range(1, 4):
                            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius + i*2, (255,255,255, 80-i*20))
                elif np.array_equal(pos, self.exit_pos):
                    pygame.draw.rect(self.screen, self.COLOR_EXIT, rect.inflate(-4, -4))

        self._draw_character(self.char1_visual_pos, self.COLOR_CHAR1)
        self._draw_character(self.char2_visual_pos, self.COLOR_CHAR2)

    def _draw_character(self, visual_pos, color):
        center_x = int(visual_pos[0] + self.GRID_SIZE / 2)
        center_y = int(visual_pos[1] + self.GRID_SIZE / 2)
        
        glow_size = int(self.GRID_SIZE * 1.2)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, color + (40,), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (center_x - glow_size // 2, center_y - glow_size // 2))

        char_size = int(self.GRID_SIZE * 0.8)
        rect = pygame.Rect(0, 0, char_size, char_size)
        rect.center = (center_x, center_y)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _render_ui(self):
        sync_percent = (self.sync_actions / self.total_moves * 100) if self.total_moves > 0 else 0
        sync_text = f"SYNC: {sync_percent:.0f}%"
        sync_surf = self.font_large.render(sync_text, True, self.COLOR_TEXT)
        self.screen.blit(sync_surf, (10, 5))

        timer_text = f"TIME: {self.time_remaining:.1f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 5))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            win = np.array_equal(self.char1_pos, self.exit_pos) and np.array_equal(self.char2_pos, self.exit_pos)
            msg = "SYNCHRONIZED!" if win else "DESYNCHRONIZED"
            msg_surf = self.font_large.render(msg, True, (255, 255, 255))
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20)))
            score_text = f"Final Score: {self.score:.2f}"
            score_surf = self.font_small.render(score_text, True, (200, 200, 200))
            self.screen.blit(score_surf, score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20)))

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps,
            "time_remaining": self.time_remaining,
            "synchronization_percent": (self.sync_actions / self.total_moves * 100) if self.total_moves > 0 else 0,
            "switches_on": sum(d['state'] for d in self.switches.values())
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display for interactive testing
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Mirror Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart or 'ESC' to quit.")
            pause = True
            while pause and running:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pause = False; running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset(); total_reward = 0; pause = False

        clock.tick(30)

    env.close()