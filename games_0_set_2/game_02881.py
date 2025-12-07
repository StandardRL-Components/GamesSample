
# Generated: 2025-08-28T06:19:29.008767
# Source Brief: brief_02881.md
# Brief Index: 2881

        
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
        "Controls: Arrow keys to move your avatar on the grid. Collect all gems before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate a grid to collect all the gems within a limited number of moves to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_W * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_H * self.CELL_SIZE) // 2
        self.NUM_GEMS = 15
        self.MAX_MOVES = 10

        # --- Colors ---
        self.COLOR_BG = (30, 30, 35)
        self.COLOR_GRID = (50, 50, 55)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (45, 45, 50, 180)  # with alpha
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (100, 100, 255), # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        # --- State Variables (initialized in reset) ---
        self.player_pos = [0, 0]
        self.gems = []
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES

        # Place player
        self.player_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        # Place gems
        self.gems = []
        possible_locations = []
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if [x, y] != self.player_pos:
                    possible_locations.append((x, y))
        
        gem_indices = self.np_random.choice(len(possible_locations), self.NUM_GEMS, replace=False)
        gem_locations = [possible_locations[i] for i in gem_indices]
        
        for i, pos in enumerate(gem_locations):
            self.gems.append({
                "pos": list(pos),
                "color": self.GEM_COLORS[i % len(self.GEM_COLORS)],
                "anim_offset": self.np_random.random() * 2 * math.pi
            })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        moved = False
        
        if movement != 0:
            target_pos = self.player_pos.copy()
            if movement == 1:  # Up
                target_pos[1] -= 1
            elif movement == 2:  # Down
                target_pos[1] += 1
            elif movement == 3:  # Left
                target_pos[0] -= 1
            elif movement == 4:  # Right
                target_pos[0] += 1

            if 0 <= target_pos[0] < self.GRID_W and 0 <= target_pos[1] < self.GRID_H:
                self.player_pos = target_pos
                moved = True

        if moved or movement != 0: # Consume a move even if it's invalid
            self.moves_left -= 1
            # SFX placeholder: # play_move_sound()
            
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem["pos"]:
                gem_to_remove = gem
                break
        
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            self.score += 1
            reward += 1
            # SFX placeholder: # play_gem_collect_sound()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and len(self.gems) == 0:
            reward += 10 # Win bonus
            # SFX placeholder: # play_win_sound()
        elif terminated and self.moves_left <= 0:
            # SFX placeholder: # play_lose_sound()
            pass
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.moves_left <= 0 or len(self.gems) == 0:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_gems()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gems_left": len(self.gems),
        }

    def _render_grid(self):
        for x in range(self.GRID_W + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_H * self.CELL_SIZE), 1)
        for y in range(self.GRID_H + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_W * self.CELL_SIZE, py), 1)

    def _render_gems(self):
        for gem in self.gems:
            center_x = self.GRID_OFFSET_X + int((gem["pos"][0] + 0.5) * self.CELL_SIZE)
            center_y = self.GRID_OFFSET_Y + int((gem["pos"][1] + 0.5) * self.CELL_SIZE)
            
            anim_phase = (self.steps * 0.2 + gem["anim_offset"])
            pulse = (math.sin(anim_phase) + 1) / 2
            
            radius = int(self.CELL_SIZE * 0.25)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, gem["color"])
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, gem["color"])
            
            glow_radius = radius + int(pulse * 5)
            glow_alpha = int(100 - pulse * 80)
            glow_color = (*gem["color"], glow_alpha)
            temp_surface = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, glow_radius, glow_radius, glow_radius, glow_color)
            pygame.gfxdraw.aacircle(temp_surface, glow_radius, glow_radius, glow_radius, glow_color)
            self.screen.blit(temp_surface, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        player_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.player_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.player_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        
        center_x, center_y = player_rect.centerx, player_rect.centery
        glow_radius = int(self.CELL_SIZE * 0.7)
        temp_surface = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        glow_color = (*self.COLOR_PLAYER, 30)
        pygame.gfxdraw.filled_circle(temp_surface, glow_radius, glow_radius, glow_radius, glow_color)
        self.screen.blit(temp_surface, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        inner_padding = int(self.CELL_SIZE * 0.15)
        inner_rect = player_rect.inflate(-inner_padding*2, -inner_padding*2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, inner_rect, border_radius=4)
        
    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        moves_text = self.font_ui.render(f"Moves Left: {max(0, self.moves_left)}", True, self.COLOR_UI_TEXT)
        gems_text = self.font_ui.render(f"Gems Left: {len(self.gems)}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (15, 10))
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 15, 10))
        self.screen.blit(gems_text, (self.WIDTH // 2 - gems_text.get_width() // 2, 10))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_UI_BG)
        self.screen.blit(overlay, (0, 0))
        
        if len(self.gems) == 0:
            message = "YOU WIN!"
            color = self.GEM_COLORS[1]
        else:
            message = "GAME OVER"
            color = self.GEM_COLORS[0]

        text_surface = self.font_big.render(message, True, color)
        text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
        self.screen.blit(text_surface, text_rect)

        final_score_text = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
        final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 40))
        self.screen.blit(final_score_text, final_score_rect)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset()
    
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    game_over = False

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over:
                acted = False
                if event.key == pygame.K_UP:
                    action[0] = 1
                    acted = True
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                    acted = True
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                    acted = True
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                    acted = True
                
                if acted:
                    obs, reward, terminated, truncated, info = env.step(action)
                    game_over = terminated
                    print(f"Action: {action}, Reward: {reward}, Info: {info}")

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over = False
                print("\n--- Game Reset ---")
                
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

    env.close()