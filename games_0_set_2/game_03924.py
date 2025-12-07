
# Generated: 2025-08-28T00:52:09.640722
# Source Brief: brief_03924.md
# Brief Index: 3924

        
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

    user_guide = (
        "Controls: ←→ to move the falling block. Press space to drop it quickly."
    )

    game_description = (
        "Stack blocks to build the tallest tower you can without it collapsing. Reach a height of 10 to win!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    TARGET_HEIGHT_UNITS = 10
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GROUND = (60, 70, 80)
    COLOR_TARGET_LINE = (0, 255, 128, 100)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_GAMEOVER = (255, 80, 80)
    COLOR_WIN = (80, 255, 80)

    # Physics & Gameplay
    BLOCK_WIDTH = 80
    BLOCK_HEIGHT = 20
    BASE_FALL_SPEED = 1.0
    GRAVITY = 0.03
    NUDGE_SPEED = 2.0
    DROP_SPEED = 8.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28, bold=True)
            self.font_big = pygame.font.SysFont(None, 52, bold=True)

        self.render_mode = render_mode
        self.game_state = {}
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = {
            "steps": 0,
            "score": 0,
            "game_over": False,
            "win": False,
            "stacked_blocks": [],
            "falling_block": None,
            "particles": [],
            "tower_height_units": 0,
            "fall_speed": self.BASE_FALL_SPEED,
            "placements": 0,
            "last_space_held": False,
            "reward_this_step": 0.0,
        }

        ground_block = {
            "rect": pygame.Rect(
                -self.WIDTH, self.HEIGHT - self.BLOCK_HEIGHT, self.WIDTH * 3, self.BLOCK_HEIGHT
            ),
            "color": self.COLOR_GROUND,
        }
        self.game_state["stacked_blocks"].append(ground_block)

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.game_state["reward_this_step"] = 0.0

        if not self.game_state["game_over"]:
            self._handle_input(action)
            self._update_physics()
            self._check_and_handle_landing()
        
        self._update_particles()
        
        self.game_state["steps"] += 1
        
        terminated = self.game_state["game_over"] or self.game_state["steps"] >= self.MAX_STEPS
        if terminated and not self.game_state["game_over"]: # Terminated by steps
            self.game_state["reward_this_step"] -= 10 # Penalty for running out of time
            self.game_state["game_over"] = True
            
        reward = self.game_state["reward_this_step"]
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        block = self.game_state["falling_block"]
        if not block:
            return

        # Horizontal movement
        if movement == 3:  # Left
            block["x_vel"] = -self.NUDGE_SPEED
        elif movement == 4:  # Right
            block["x_vel"] = self.NUDGE_SPEED
        else:
            block["x_vel"] = 0

        # Drop action
        space_pressed_now = space_held == 1 and not self.game_state["last_space_held"]
        if space_pressed_now:
            block["y_vel"] = max(block["y_vel"], self.DROP_SPEED)
            # Sound placeholder: pygame.mixer.Sound("drop.wav").play()

        self.game_state["last_space_held"] = (space_held == 1)

    def _update_physics(self):
        block = self.game_state["falling_block"]
        if not block:
            return

        # Apply velocity and gravity
        block["y_vel"] += self.GRAVITY
        block["x"] += block["x_vel"]
        block["y"] += block["y_vel"]

        # Keep block within horizontal bounds
        block["x"] = np.clip(block["x"], self.BLOCK_WIDTH / 2, self.WIDTH - self.BLOCK_WIDTH / 2)
        block["rect"].center = (int(block["x"]), int(block["y"]))

    def _check_and_handle_landing(self):
        block = self.game_state["falling_block"]
        if not block:
            return

        support_block = None
        for stacked in self.game_state["stacked_blocks"]:
            if block["rect"].colliderect(stacked["rect"]):
                # Find the highest block we are colliding with
                if support_block is None or stacked["rect"].top < support_block["rect"].top:
                    support_block = stacked

        if support_block:
            # --- LANDING LOGIC ---
            landed_block = block
            self.game_state["falling_block"] = None
            
            # Snap position
            landed_block["rect"].bottom = support_block["rect"].top
            landed_block["y"] = landed_block["rect"].centery
            
            # Add particles on impact
            self._add_particles(
                landed_block["rect"].centerx,
                landed_block["rect"].bottom,
                landed_block["color"],
                count=10
            )
            # Sound placeholder: pygame.mixer.Sound("thud.wav").play()

            # Placement reward
            self.game_state["reward_this_step"] += 0.1
            self.game_state["placements"] += 1

            # Check stability
            offset = abs(landed_block["rect"].centerx - support_block["rect"].centerx)
            is_stable = offset <= support_block["rect"].width / 2

            if not is_stable:
                self.game_state["game_over"] = True
                self.game_state["reward_this_step"] -= 50
                # Sound placeholder: pygame.mixer.Sound("collapse.wav").play()
                self._add_particles(
                    landed_block["rect"].centerx,
                    landed_block["rect"].bottom,
                    self.COLOR_GAMEOVER,
                    count=50,
                    lifespan=90
                )
            else:
                self.game_state["stacked_blocks"].append(landed_block)
                
                old_height = self.game_state["tower_height_units"]
                self._calculate_tower_height()
                new_height = self.game_state["tower_height_units"]
                
                height_gain = new_height - old_height
                if height_gain > 0:
                    self.game_state["reward_this_step"] += height_gain * 1.0
                    self.game_state["score"] += height_gain
                
                # Check for win condition
                if self.game_state["tower_height_units"] >= self.TARGET_HEIGHT_UNITS:
                    self.game_state["game_over"] = True
                    self.game_state["win"] = True
                    self.game_state["reward_this_step"] += 100
                    # Sound placeholder: pygame.mixer.Sound("win.wav").play()
                else:
                    self._update_difficulty()
                    self._spawn_new_block()

    def _spawn_new_block(self):
        x_pos = self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8)
        hue = self.np_random.uniform(0, 360)
        color = pygame.Color(0)
        color.hsva = (hue, 80, 100, 100)
        
        self.game_state["falling_block"] = {
            "x": x_pos,
            "y": -self.BLOCK_HEIGHT,
            "x_vel": 0,
            "y_vel": self.game_state["fall_speed"],
            "rect": pygame.Rect(0, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "color": tuple(color)[:3],
        }

    def _update_difficulty(self):
        if self.game_state["placements"] > 0 and self.game_state["placements"] % 20 == 0:
            self.game_state["fall_speed"] += 0.05

    def _calculate_tower_height(self):
        ground_y = self.HEIGHT - self.BLOCK_HEIGHT
        min_y = ground_y
        for block in self.game_state["stacked_blocks"]:
            if block["rect"].width < self.WIDTH: # Exclude ground
                min_y = min(min_y, block["rect"].top)
        
        height_pixels = ground_y - min_y
        self.game_state["tower_height_units"] = round(height_pixels / self.BLOCK_HEIGHT)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render target height line
        target_y = self.HEIGHT - self.BLOCK_HEIGHT * (self.TARGET_HEIGHT_UNITS + 1)
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.WIDTH, target_y), 2)

        # Render all blocks
        all_blocks = self.game_state["stacked_blocks"]
        if self.game_state["falling_block"]:
            all_blocks = all_blocks + [self.game_state["falling_block"]]

        for block in all_blocks:
            self._render_block(block)

        # Render particles
        for p in self.game_state["particles"]:
            size = int(p["size"] * (p["life"] / p["lifespan"]))
            if size > 0:
                rect = pygame.Rect(int(p["x"] - size/2), int(p["y"] - size/2), size, size)
                pygame.draw.rect(self.screen, p["color"], rect)

    def _render_block(self, block_data):
        rect = block_data["rect"]
        color = block_data["color"]
        
        # Draw a slightly darker shadow/base for depth
        shadow_color = tuple(max(0, c - 20) for c in color)
        shadow_rect = rect.copy()
        shadow_rect.height = max(1, int(rect.height * 0.3))
        shadow_rect.bottom = rect.bottom
        pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_bottom_left_radius=2, border_bottom_right_radius=2)

        # Draw main block body
        main_rect = rect.copy()
        main_rect.height = int(rect.height * 0.8)
        main_rect.bottom = shadow_rect.top
        pygame.draw.rect(self.screen, color, main_rect, border_top_left_radius=2, border_top_right_radius=2)
        
        # Add a subtle highlight
        highlight_color = tuple(min(255, c + 30) for c in color)
        pygame.draw.line(self.screen, highlight_color, main_rect.topleft, main_rect.topright, 1)

    def _render_ui(self):
        height_text = f"Height: {self.game_state['tower_height_units']} / {self.TARGET_HEIGHT_UNITS}"
        self._draw_text(height_text, (10, 10), self.font_main)
        
        steps_text = f"Steps: {self.game_state['steps']} / {self.MAX_STEPS}"
        self._draw_text(steps_text, (self.WIDTH - 10, 10), self.font_main, align="right")

        if self.game_state["game_over"]:
            if self.game_state["win"]:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "TOWER COLLAPSED"
                color = self.COLOR_GAMEOVER
            
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_big, color=color, align="center")

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="left"):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        if align == "center":
            text_rect = text_surf.get_rect(center=pos)
        elif align == "right":
            text_rect = text_surf.get_rect(topright=pos)
        else: # left
            text_rect = text_surf.get_rect(topleft=pos)
            
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _add_particles(self, x, y, color, count=10, lifespan=40):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.game_state["particles"].append({
                "x": x,
                "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": lifespan,
                "lifespan": lifespan,
                "color": color,
                "size": self.np_random.uniform(3, 7)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.game_state["particles"]:
            p["life"] -= 1
            if p["life"] > 0:
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["vy"] += 0.1  # Gravity on particles
                active_particles.append(p)
        self.game_state["particles"] = active_particles

    def _get_info(self):
        return {
            "score": self.game_state["tower_height_units"],
            "steps": self.game_state["steps"],
            "win": self.game_state["win"],
        }

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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a way to render the environment and get key presses.
    # This example shows how to run the environment loop.
    
    pygame.display.set_caption("Tower Stacker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # not used
        
        action = [movement, space_held, shift_held]
        
        # --- Event handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We need to transpose it back for display.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)

    env.close()