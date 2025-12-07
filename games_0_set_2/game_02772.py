
# Generated: 2025-08-27T21:22:50.525263
# Source Brief: brief_02772.md
# Brief Index: 2772

        
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

    user_guide = (
        "Controls: Press space to break the blocks as they cross the hit line."
    )

    game_description = (
        "Fast-paced rhythm game. Tap in time with the falling blocks to break them and score points."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS = 10
        self.GRID_COL_WIDTH = self.WIDTH // self.GRID_COLS
        self.MAX_STEPS = 1500 # Increased to allow more time for 100 blocks
        self.WIN_CONDITION = 100
        self.LOSE_CONDITION = 5

        # --- Colors ---
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_GRID = (42, 42, 78)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HIT_ZONE = (0, 255, 255, 150)
        self.BLOCK_COLORS = [
            (255, 0, 128), (255, 128, 0), (0, 200, 255), (128, 255, 0)
        ]
        self.ACCURACY_COLORS = {
            "perfect": (255, 255, 0),
            "good": (0, 255, 0),
            "bad": (255, 0, 0),
        }

        # --- Game Mechanics ---
        self.HIT_ZONE_Y = 350
        self.BLOCK_HEIGHT = 20
        self.BLOCK_WIDTH = self.GRID_COL_WIDTH - 8
        self.INITIAL_FALL_SPEED = 2.0
        self.SPEED_INCREASE_INTERVAL = 25
        self.SPEED_INCREASE_AMOUNT = 0.05

        # Timing windows (in pixels from hit zone center)
        self.PERFECT_WINDOW = 2
        self.GOOD_WINDOW = 10
        self.BAD_WINDOW = 20
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Initialization ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cleared_blocks = 0
        self.missed_blocks = 0
        self.fall_speed = 0.0
        self.blocks = []
        self.particles = []
        self.effects = []
        self.last_action_frame = -1

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cleared_blocks = 0
        self.missed_blocks = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_action_frame = -1

        self.blocks.clear()
        self.particles.clear()
        self.effects.clear()

        self._spawn_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        # --- Handle Input ---
        space_pressed = action[1] == 1
        
        # Prevent holding space from triggering multiple times
        if space_pressed and self.steps > self.last_action_frame:
            reward += self._handle_hit()
            self.last_action_frame = self.steps

        # --- Update Game State ---
        self._update_blocks()
        self._update_particles()
        self._update_effects()

        # --- Check for Misses ---
        if self.blocks and self.blocks[0]["pos"][1] > self.HEIGHT:
            self.missed_blocks += 1
            self.blocks.pop(0)
            self._spawn_block()
            # Sound: Miss sound
            self.effects.append({
                "type": "text", "pos": [self.WIDTH // 2, self.HEIGHT // 2],
                "text": "MISS", "color": self.ACCURACY_COLORS["bad"], "life": 30
            })


        # --- Check Termination Conditions ---
        if self.cleared_blocks >= self.WIN_CONDITION:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.missed_blocks >= self.LOSE_CONDITION:
            reward -= 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_hit(self):
        if not self.blocks:
            return 0.0

        block = self.blocks[0]
        block_center_y = block["pos"][1] + self.BLOCK_HEIGHT / 2
        distance = abs(block_center_y - self.HIT_ZONE_Y)
        
        hit_pos = [block["pos"][0] + self.BLOCK_WIDTH / 2, self.HIT_ZONE_Y]

        if distance <= self.BAD_WINDOW:
            # Sound: Block break sound
            self.cleared_blocks += 1
            self._create_particles(block["pos"], block["color"])
            self.blocks.pop(0)
            self._spawn_block()
            self._update_difficulty()

            if distance <= self.PERFECT_WINDOW:
                self._create_hit_effect(hit_pos, "perfect")
                self.effects.append({
                    "type": "text", "pos": [hit_pos[0], hit_pos[1] - 30],
                    "text": "PERFECT!", "color": self.ACCURACY_COLORS["perfect"], "life": 40
                })
                self.score += 15
                return 15.0
            elif distance <= self.GOOD_WINDOW:
                self._create_hit_effect(hit_pos, "good")
                self.score += 5
                return 5.0
            else: # Bad hit
                self._create_hit_effect(hit_pos, "bad")
                self.score -= 1
                return -1.0
        
        return 0.0 # No hit

    def _update_difficulty(self):
        level = self.cleared_blocks // self.SPEED_INCREASE_INTERVAL
        self.fall_speed = self.INITIAL_FALL_SPEED + level * self.SPEED_INCREASE_AMOUNT

    def _update_blocks(self):
        for block in self.blocks:
            block["pos"][1] += self.fall_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_effects(self):
        for e in self.effects[:]:
            e["life"] -= 1
            if e["type"] == "circle":
                e["radius"] += e["expand_rate"]
            if e["life"] <= 0:
                self.effects.remove(e)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            x = i * self.GRID_COL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)

        # Draw hit zone
        hit_zone_rect = pygame.Rect(0, self.HIT_ZONE_Y - 2, self.WIDTH, 4)
        shape_surf = pygame.Surface(hit_zone_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, self.COLOR_HIT_ZONE, shape_surf.get_rect())
        self.screen.blit(shape_surf, hit_zone_rect.topleft)

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], 
                             pygame.Rect(int(block["pos"][0]), int(block["pos"][1]), 
                                         self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                             border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            p_color = (*p["color"], alpha)
            p_surf = pygame.Surface((p["size"], p["size"]), pygame.SRCALPHA)
            p_surf.fill(p_color)
            self.screen.blit(p_surf, (int(p["pos"][0]), int(p["pos"][1])))

        # Draw effects
        for e in self.effects:
            alpha = int(max(0, 255 * (e["life"] / 30)))
            if e["type"] == "circle":
                color = (*e["color"], alpha // 2)
                pos = (int(e["pos"][0]), int(e["pos"][1]))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(e["radius"]), color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(e["radius"]-1), color)
            elif e["type"] == "text":
                alpha = int(max(0, 255 * (e["life"] / 40)))
                self._draw_text(e["text"], e["pos"], self.font_feedback, e["color"], alpha, center=True)


    def _render_ui(self):
        self._draw_text(f"SCORE: {self.score}", (10, 10), self.font_ui, self.COLOR_TEXT)
        self._draw_text(f"MISSES: {self.missed_blocks}/{self.LOSE_CONDITION}", (self.WIDTH - 160, 10), self.font_ui, self.COLOR_TEXT)
        
        cleared_text = f"CLEARED: {self.cleared_blocks}/{self.WIN_CONDITION}"
        text_width = self.font_ui.size(cleared_text)[0]
        self._draw_text(cleared_text, (self.WIDTH/2 - text_width/2, self.HEIGHT - 35), self.font_ui, self.COLOR_TEXT)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.cleared_blocks >= self.WIN_CONDITION:
                msg = "YOU WIN!"
                color = self.ACCURACY_COLORS["good"]
            else:
                msg = "GAME OVER"
                color = self.ACCURACY_COLORS["bad"]
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_game_over, color, center=True)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cleared_blocks": self.cleared_blocks,
            "missed_blocks": self.missed_blocks,
        }

    def _spawn_block(self):
        col = self.np_random.integers(0, self.GRID_COLS)
        x_pos = col * self.GRID_COL_WIDTH + (self.GRID_COL_WIDTH - self.BLOCK_WIDTH) / 2
        y_pos = -self.BLOCK_HEIGHT
        color = self.np_random.choice(self.BLOCK_COLORS)
        
        self.blocks.append({
            "pos": [x_pos, y_pos],
            "color": color,
        })

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": [pos[0] + self.BLOCK_WIDTH/2, pos[1] + self.BLOCK_HEIGHT/2],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": life,
                "max_life": life,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _create_hit_effect(self, pos, accuracy):
        self.effects.append({
            "type": "circle",
            "pos": pos,
            "radius": 10,
            "expand_rate": 2,
            "life": 30,
            "color": self.ACCURACY_COLORS[accuracy]
        })

    def _draw_text(self, text, pos, font, color, alpha=255, center=False):
        text_surface = font.render(text, True, color)
        text_surface.set_alpha(alpha)
        if center:
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, pos)

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
    pygame.display.set_caption("Rhythm Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        space_pressed = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed = 1
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    total_reward = 0

        # Action format: [movement, space, shift]
        action = [0, space_pressed, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()