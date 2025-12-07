import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:32:34.769328
# Source Brief: brief_01749.md
# Brief Index: 1749
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Dice Racer Environment:
    Roll a dice to determine how many high-multiplier paths you can choose from.
    Race your pawn to the end of any path before time runs out. Triggering
    chain reactions by rolling the same number twice gives a speed boost.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Race your pawn to the end of a path by rolling a die. Higher rolls unlock higher-scoring paths, "
        "and consecutive identical rolls grant a speed boost."
    )
    user_guide = (
        "Press space to roll the die. Use arrow keys (e.g., ↑ for path 1) to select an available path. "
        "Hold Shift + ↑ to select the 5th path."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Colors & Style ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_PAWN = (255, 255, 100)
        self.COLOR_PAWN_GLOW = (255, 255, 100, 50)
        self.COLOR_START_FINISH = (200, 200, 220)
        self.PATH_COLORS = [
            (0, 200, 100),   # Green (x1)
            (0, 150, 255),   # Blue (x2)
            (255, 200, 0),   # Yellow (x3)
            (255, 120, 0),   # Orange (x4)
            (255, 50, 50),    # Red (x5)
        ]
        self.PATH_COLORS_DIM = [(c[0]*0.4, c[1]*0.4, c[2]*0.4) for c in self.PATH_COLORS]
        self.PATH_COLORS_HIGHLIGHT = [(min(255, c[0]*1.5), min(255, c[1]*1.5), min(255, c[2]*1.5)) for c in self.PATH_COLORS]

        # --- Game Constants ---
        self.MAX_TIME = 45.0
        self.MAX_STEPS = 1350 # 45s * 30fps
        self.NUM_PATHS = 5
        self.PATH_SEGMENTS = 60
        self.START_POS = (80, self.HEIGHT // 2)
        self.FINISH_X = self.WIDTH - 80

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0.0
        self.game_phase = "IDLE" # IDLE, ROLLING, CHOOSING, MOVING, GAME_OVER
        
        self.pawn_path_idx = -1
        self.pawn_segment_idx = 0
        self.pawn_visual_pos = self.START_POS
        
        self.dice_roll = 0
        self.last_dice_roll = 0
        self.dice_anim_timer = 0
        
        self.paths = []
        self.particles = []
        
        self.last_space_held = False
        
        self._generate_paths()
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def _generate_paths(self):
        """Pre-calculates the points for each path using Bezier curves."""
        self.paths = []
        y_spread = 150
        for i in range(self.NUM_PATHS):
            path = {
                "multiplier": i + 1,
                "color": self.PATH_COLORS[i],
                "color_dim": self.PATH_COLORS_DIM[i],
                "color_highlight": self.PATH_COLORS_HIGHLIGHT[i],
                "points": [],
                "chain_active": False,
            }
            
            start_y = self.START_POS[1]
            end_y = self.START_POS[1] + (i - (self.NUM_PATHS-1)/2) * (y_spread / (self.NUM_PATHS-1))
            
            # Control point for the curve
            ctrl_x = self.START_POS[0] + (self.FINISH_X - self.START_POS[0]) * 0.5
            ctrl_y = start_y + (end_y - start_y) * random.uniform(0.3, 0.7)

            for t_step in range(self.PATH_SEGMENTS + 1):
                t = t_step / self.PATH_SEGMENTS
                inv_t = 1 - t
                
                x = (inv_t**2 * self.START_POS[0]) + (2 * inv_t * t * ctrl_x) + (t**2 * self.FINISH_X)
                y = (inv_t**2 * start_y) + (2 * inv_t * t * ctrl_y) + (t**2 * end_y)
                path["points"].append((x, y))
            
            self.paths.append(path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        self.game_phase = "IDLE"
        
        self.pawn_path_idx = -1
        self.pawn_segment_idx = 0
        self.pawn_visual_pos = self.START_POS
        
        self.dice_roll = 0
        self.last_dice_roll = 0
        self.dice_anim_timer = 0
        self.move_segments_remaining = 0
        
        for path in self.paths:
            path["chain_active"] = False
            
        self.particles.clear()
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        if self.game_phase == "GAME_OVER":
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        # --- Unpack Actions ---
        movement_action = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Update Game State Machine ---
        if self.game_phase == "IDLE":
            if space_pressed:
                # SFX: Dice shake
                self.game_phase = "ROLLING"
                self.dice_anim_timer = 15 # 0.5s at 30fps
        
        elif self.game_phase == "ROLLING":
            self.dice_anim_timer -= 1
            if self.dice_anim_timer <= 0:
                self.last_dice_roll = self.dice_roll
                self.dice_roll = self.np_random.integers(1, 7)
                # SFX: Dice land
                self.game_phase = "CHOOSING"

        elif self.game_phase == "CHOOSING":
            chosen_path = -1
            if movement_action > 0:
                if shift_held and movement_action == 1: # Shift + Up for Path 5
                    chosen_path = 4
                elif not shift_held and 1 <= movement_action <= 4:
                    chosen_path = movement_action - 1
            
            if chosen_path != -1 and chosen_path < self.dice_roll:
                self.pawn_path_idx = chosen_path
                self.pawn_segment_idx = 0
                self.move_segments_remaining = self.dice_roll
                
                # Check for chain reaction
                if self.dice_roll == self.last_dice_roll and self.dice_roll > 0:
                    # SFX: Chain reaction trigger
                    path_to_chain = self.paths[self.dice_roll - 1]
                    path_to_chain["chain_active"] = True
                    reward += 1.0
                    self._create_particles(self.START_POS, path_to_chain['color'], 30)

                self.game_phase = "MOVING"
        
        elif self.game_phase == "MOVING":
            path = self.paths[self.pawn_path_idx]
            speed_multiplier = 2.0 if path["chain_active"] else 1.0
            
            target_pos = path["points"][self.pawn_segment_idx + 1]
            
            # Interpolate pawn position
            dx = target_pos[0] - self.pawn_visual_pos[0]
            dy = target_pos[1] - self.pawn_visual_pos[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            move_speed = 10.0 * speed_multiplier
            if dist < move_speed:
                self.pawn_visual_pos = target_pos
                self.pawn_segment_idx += 1
                self.move_segments_remaining -= 1
                
                # SFX: Pawn move tick
                self.score += path["multiplier"]
                reward += 0.1
                
                if self.np_random.random() < 0.5:
                    self._create_particles(self.pawn_visual_pos, self.COLOR_PAWN, 1, life=10, speed=1)

                if self.pawn_segment_idx >= self.PATH_SEGMENTS:
                    self.game_phase = "GAME_OVER"
                    self.game_over = True
                    reward += 100.0
                    # SFX: Win fanfare
                    self._create_particles(self.pawn_visual_pos, (255,255,255), 100)
                elif self.move_segments_remaining <= 0:
                    self.game_phase = "IDLE"
                    path["chain_active"] = False # Chain lasts for one turn
            else:
                self.pawn_visual_pos = (
                    self.pawn_visual_pos[0] + (dx / dist) * move_speed,
                    self.pawn_visual_pos[1] + (dy / dist) * move_speed,
                )

        # --- Global Updates ---
        self._update_particles()
        self.time_remaining -= 1.0 / 30.0 # Assuming 30 FPS for env stepping
        
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True
            self.game_phase = "GAME_OVER"
            reward -= 100.0
            # SFX: Lose buzzer
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] += 0.05 # Gravity

    def _create_particles(self, pos, color, count, life=30, speed=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * s, math.sin(angle) * s],
                'life': self.np_random.integers(life // 2, life + 1),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (self.START_POS[0], 50), (self.START_POS[0], self.HEIGHT - 50), 3)
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (self.FINISH_X, 50), (self.FINISH_X, self.HEIGHT - 50), 3)

        # Draw paths
        for i, path in enumerate(self.paths):
            is_available = self.game_phase == "CHOOSING" and i < self.dice_roll
            color = path["color_highlight"] if is_available else path["color"] if path["chain_active"] else path["color_dim"]
            width = 4 if path["chain_active"] else 2
            
            pygame.draw.aalines(self.screen, color, False, path["points"], 1)
            if is_available or path["chain_active"]:
                 for p_idx in range(len(path["points"])-1):
                     pygame.draw.line(self.screen, color, path["points"][p_idx], path["points"][p_idx+1], width)

            # Draw multiplier text
            text_surf = self.font_medium.render(f"x{path['multiplier']}", True, color)
            text_rect = text_surf.get_rect(center=(path["points"][self.PATH_SEGMENTS//3]))
            self.screen.blit(text_surf, text_rect)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Draw pawn
        if self.pawn_path_idx != -1 or self.game_phase != "IDLE":
            pos = (int(self.pawn_visual_pos[0]), int(self.pawn_visual_pos[1]))
            # Glow effect
            for i in range(4, 0, -1):
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8 + i*2, self.COLOR_PAWN_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_PAWN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, (255,255,255))
    
    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Timer
        time_color = (255, 100, 100) if self.time_remaining < 10 else self.COLOR_UI_TEXT
        time_surf = self.font_medium.render(f"Time: {max(0, self.time_remaining):.1f}", True, time_color)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(time_surf, time_rect)

        # Dice display
        dice_text = str(self.dice_roll)
        if self.game_phase == "ROLLING":
            dice_text = str(self.np_random.integers(1, 7))
        
        if self.dice_roll > 0 or self.game_phase == "ROLLING":
            dice_surf = self.font_big.render(dice_text, True, self.COLOR_UI_TEXT)
            dice_rect = dice_surf.get_rect(center=(self.WIDTH // 2, 40))
            pygame.draw.rect(self.screen, (50, 60, 80), dice_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, dice_rect.inflate(20, 10), 2)
            self.screen.blit(dice_surf, dice_rect)

        # Game phase instructions
        msg = ""
        if self.game_phase == "IDLE" and self.pawn_path_idx == -1:
            msg = "Press SPACE to Roll"
        elif self.game_phase == "CHOOSING":
            msg = f"Choose Path (1-{self.dice_roll})"
        elif self.game_phase == "GAME_OVER":
            msg = "GAME OVER"
        
        if msg:
            msg_surf = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 40))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "dice_roll": self.dice_roll,
            "game_phase": self.game_phase
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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


# Example of how to run the environment
if __name__ == '__main__':
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This block allows a human to play the game.
    # Controls:
    # Arrow Keys: Choose Path (Up for 1, Down for 2, Left for 3, Right for 4)
    # Shift + Up: Choose Path 5
    # Space: Roll dice
    # Q: Quit
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Dice Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = env.action_space.sample() # Default to random
        action[0] = 0 # No movement
        action[1] = 0 # Space not held
        action[2] = 0 # Shift not held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        if keys[pygame.K_DOWN]: action[0] = 2
        if keys[pygame.K_LEFT]: action[0] = 3
        if keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            env.reset()

        env.clock.tick(30)

    env.close()