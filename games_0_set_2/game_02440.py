import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to select a floor type to build. Use ↑↓ to select an existing floor or the top build slot. "
        "Press Space to build the selected type. Press Shift to upgrade the selected floor."
    )

    game_description = (
        "Build and manage a tiny isometric tower to maximize profits. "
        "Build different floor types, upgrade them, and watch your fortune grow. "
        "Reach $100,000 to win, but don't go bankrupt!"
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 35, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (45, 55, 80)
    COLOR_GREEN = (60, 220, 120)
    COLOR_RED = (220, 60, 60)
    COLOR_GOLD = (255, 215, 0)
    COLOR_SELECTOR = (255, 255, 255, 150)

    # Floor Types
    FLOOR_TYPES = {
        "RESIDENTIAL": {"color": (52, 152, 219), "name": "APT", "income": 5, "cost": 100},
        "COMMERCIAL": {"color": (231, 76, 60), "name": "SHOP", "income": 15, "cost": 250},
        "FOOD": {"color": (46, 204, 113), "name": "FOOD", "income": 25, "cost": 400},
        "RECREATION": {"color": (155, 89, 182), "name": "FUN", "income": 40, "cost": 600},
    }
    FLOOR_TYPE_KEYS = list(FLOOR_TYPES.keys())

    # Game Parameters
    WIDTH, HEIGHT = 640, 400
    WIN_BALANCE = 100_000
    START_BALANCE = 1_000
    MAX_STEPS = 1000
    BASE_UPGRADE_COST = 150

    # Isometric Rendering
    ISO_TILE_W = 96
    ISO_TILE_H = 48
    ISO_FLOOR_H = 32

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.ui_font = pygame.font.Font(None, 28)
        self.floor_font = pygame.font.Font(None, 18)
        self.particle_font = pygame.font.Font(None, 22)

        self.origin_x = self.WIDTH // 2
        self.origin_y = 80
        
        # Note: reset() is called by the gym.make wrapper, so not explicitly needed here
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.START_BALANCE
        self.floors = []
        self.steps = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.selector_pos = 0  # 0 is the build slot, 1+ are floors
        self.build_selector_idx = 0
        
        self.particles = []
        self.construction_anim = None # {'floor_idx': int, 'timer': int}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self.steps += 1
        
        self._update_animations()
        self._handle_input(action)
        self._collect_income()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        reward = self.reward_this_step
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        # --- Selector Movement ---
        if movement == 1: # Up
            self.selector_pos = max(0, self.selector_pos - 1)
        elif movement == 2: # Down
            self.selector_pos = min(len(self.floors), self.selector_pos + 1)
        elif movement == 3: # Left
            self.build_selector_idx = (self.build_selector_idx - 1) % len(self.FLOOR_TYPE_KEYS)
        elif movement == 4: # Right
            self.build_selector_idx = (self.build_selector_idx + 1) % len(self.FLOOR_TYPE_KEYS)
            
        # --- Build Action ---
        if space_press and self.selector_pos == 0:
            self._build_floor()
            
        # --- Upgrade Action ---
        if shift_press and self.selector_pos > 0:
            self._upgrade_floor()

    def _get_cost(self, base_cost):
        cost_multiplier = 1 + 0.1 * math.floor(max(0, self.balance) / 10000)
        return base_cost * cost_multiplier

    def _build_floor(self):
        floor_key = self.FLOOR_TYPE_KEYS[self.build_selector_idx]
        cost = self._get_cost(self.FLOOR_TYPES[floor_key]["cost"])
        
        if self.balance >= cost:
            self.balance -= cost
            new_floor = {"type": floor_key, "level": 1, "timer": 0}
            self.floors.insert(0, new_floor)
            self.selector_pos = 1 # Move selector to the new floor
            self.reward_this_step += 1.0
            self.construction_anim = {'floor_idx': 0, 'timer': 15} # Animate for 15 frames
        else:
            self.reward_this_step -= 0.1
            self._spawn_particle("- INSUFFICIENT FUNDS -", (self.WIDTH//2, 50), self.COLOR_RED, 30)

    def _upgrade_floor(self):
        floor_idx = self.selector_pos - 1
        if 0 <= floor_idx < len(self.floors):
            floor = self.floors[floor_idx]
            cost = self._get_cost(self.BASE_UPGRADE_COST * floor['level'] * 1.5)
            
            if self.balance >= cost:
                self.balance -= cost
                floor['level'] += 1
                self.reward_this_step += 2.0
                
                floor_screen_pos = self._get_floor_screen_pos(floor_idx)
                self._spawn_particle("UPGRADE!", (floor_screen_pos[0], floor_screen_pos[1] - 20), self.COLOR_GOLD, 30)
            else:
                self.reward_this_step -= 0.1

    def _collect_income(self):
        total_income_this_step = 0
        for floor in self.floors:
            floor_info = self.FLOOR_TYPES[floor['type']]
            income = floor_info['income'] * floor['level']
            
            num_res_floors = sum(1 for f in self.floors if f['type'] == 'RESIDENTIAL')
            if floor['type'] != 'RESIDENTIAL' and num_res_floors > 0:
                income *= (1 + num_res_floors * 0.1)
                
            total_income_this_step += income
        
        if total_income_this_step > 0:
            self.balance += total_income_this_step
            self.reward_this_step += (total_income_this_step / 100) * 0.1
            
            if self.steps % 5 == 0 and total_income_this_step > 1:
                self._spawn_particle(f"+${int(total_income_this_step)}", (self.WIDTH - 100, 25), self.COLOR_GREEN, 20)

    def _check_termination(self):
        if self.balance >= self.WIN_BALANCE:
            self.game_over = True
            self.reward_this_step += 100.0
            return True
        if self.balance < 0:
            self.game_over = True
            self.reward_this_step -= 100.0
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.balance,
            "steps": self.steps,
            "floors": len(self.floors),
        }

    # --- Rendering Methods ---
    
    def _draw_iso_cube(self, surface, pos, color, height):
        x, y = pos
        w, h = self.ISO_TILE_W, self.ISO_TILE_H
        
        # Points for the base
        p_bottom = (x, y)
        p_left = (x - w / 2, y + h / 2)
        p_right = (x + w / 2, y + h / 2)
        p_top = (x, y + h)

        # Draw side faces
        left_face = [(p_left[0], p_left[1]), (p_left[0], p_left[1] - height), (p_bottom[0], p_bottom[1] - height), (p_bottom[0], p_bottom[1])]
        right_face = [(p_right[0], p_right[1]), (p_right[0], p_right[1] - height), (p_top[0], p_top[1] - height), (p_top[0], p_top[1])]
        
        darker_color = tuple(max(0, int(c * 0.7)) for c in color)
        pygame.draw.polygon(surface, darker_color, left_face)
        pygame.draw.polygon(surface, darker_color, right_face)
        pygame.gfxdraw.aapolygon(surface, left_face, darker_color)
        pygame.gfxdraw.aapolygon(surface, right_face, darker_color)

        # Draw top face
        top_face = [(p_bottom[0], p_bottom[1] - height), (p_left[0], p_left[1] - height), (p_top[0], p_top[1] - height), (p_right[0], p_right[1] - height)]
        pygame.draw.polygon(surface, color, top_face)
        pygame.gfxdraw.aapolygon(surface, top_face, color)

    def _get_floor_screen_pos(self, floor_idx):
        return (self.origin_x, self.origin_y + self.ISO_TILE_H + floor_idx * self.ISO_FLOOR_H)

    def _render_game(self):
        # Render tower from bottom up
        for i in range(len(self.floors) - 1, -1, -1):
            floor = self.floors[i]
            pos = self._get_floor_screen_pos(i)
            
            if self.construction_anim and self.construction_anim['floor_idx'] == i:
                scaffold_color = (180, 180, 100)
                self._draw_iso_cube(self.screen, pos, scaffold_color, self.ISO_FLOOR_H)
            else:
                floor_info = self.FLOOR_TYPES[floor['type']]
                self._draw_iso_cube(self.screen, pos, floor_info['color'], self.ISO_FLOOR_H)

                text_surf = self.floor_font.render(f"{floor_info['name']} L{floor['level']}", True, self.COLOR_UI_TEXT)
                text_rect = text_surf.get_rect(center=(pos[0], pos[1] - self.ISO_FLOOR_H / 2))
                self.screen.blit(text_surf, text_rect)
        
        self._render_selector()
        self._render_particles()

    def _render_selector(self):
        if self.selector_pos == 0: # Build slot
            pos = (self.origin_x, self.origin_y)
            height = self.ISO_FLOOR_H
        else: # Existing floor
            floor_idx = self.selector_pos - 1
            pos = self._get_floor_screen_pos(floor_idx)
            height = self.ISO_FLOOR_H

        w, h = self.ISO_TILE_W, self.ISO_TILE_H
        
        # Points for the top face of the cube
        points = [
            (pos[0], pos[1] - height),
            (pos[0] - w/2, pos[1] - height + h/2),
            (pos[0], pos[1] - height + h),
            (pos[0] + w/2, pos[1] - height + h/2)
        ]
        
        alpha = 128 + 64 * math.sin(pygame.time.get_ticks() * 0.005)
        color = (*self.COLOR_SELECTOR[:3], int(alpha))
        
        temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.lines(temp_surf, color, True, points, 3)
        self.screen.blit(temp_surf, (0,0))


    def _render_ui(self):
        bar_rect = pygame.Rect(0, 0, self.WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, bar_rect)
        
        balance_text = f"${int(self.balance):,}"
        balance_color = self.COLOR_GREEN if self.balance >= 0 else self.COLOR_RED
        balance_surf = self.ui_font.render(balance_text, True, balance_color)
        balance_rect = balance_surf.get_rect(center=(self.WIDTH // 2, 20))
        self.screen.blit(balance_surf, balance_rect)
        
        steps_text = f"Step: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.ui_font.render(steps_text, True, self.COLOR_UI_TEXT)
        steps_rect = steps_surf.get_rect(midleft=(10, 20))
        self.screen.blit(steps_surf, steps_rect)
        
        if self.selector_pos == 0:
            selected_key = self.FLOOR_TYPE_KEYS[self.build_selector_idx]
            floor_info = self.FLOOR_TYPES[selected_key]
            cost = self._get_cost(floor_info['cost'])
            
            build_text = f"Build: {floor_info['name']} (${int(cost)})"
            build_surf = self.ui_font.render(build_text, True, floor_info['color'])
            build_rect = build_surf.get_rect(center=(self.origin_x, self.origin_y - 25))
            self.screen.blit(build_surf, build_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.balance >= self.WIN_BALANCE:
                msg = "YOU WIN!"
                color = self.COLOR_GOLD
            else:
                msg = "GAME OVER"
                color = self.COLOR_RED
            
            end_font = pygame.font.Font(None, 72)
            end_surf = end_font.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            
            overlay.blit(end_surf, end_rect)
            self.screen.blit(overlay, (0,0))

    def _spawn_particle(self, text, pos, color, lifetime):
        self.particles.append({'text': text, 'pos': list(pos), 'color': color, 'life': lifetime, 'max_life': lifetime})

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            try:
                # Create a temporary surface for the text to apply alpha correctly
                temp_surf = self.particle_font.render(p['text'], True, p['color']).convert_alpha()
                temp_surf.set_alpha(alpha)
                text_rect = temp_surf.get_rect(center=p['pos'])
                self.screen.blit(temp_surf, text_rect)
            except pygame.error:
                # Fallback for potential font rendering issues in some headless setups
                pass
    
    def _update_animations(self):
        for p in self.particles[:]:
            p['life'] -= 1
            p['pos'][1] -= 0.5
            if p['life'] <= 0:
                self.particles.remove(p)
        
        if self.construction_anim:
            self.construction_anim['timer'] -= 1
            if self.construction_anim['timer'] <= 0:
                self.construction_anim = None


if __name__ == "__main__":
    # This block allows you to run the environment directly for testing.
    # It's recommended to use a proper agent or a manual control loop for interaction.
    
    # --- Manual Control Example ---
    # To play manually, you need a display. Comment out the headless environment variable at the top.
    # Then uncomment this block.
    
    # os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    # import pygame
    #
    # env = GameEnv()
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # pygame.display.set_caption("Tower Tycoon")
    # clock = pygame.time.Clock()
    # running = True
    #
    # print("\n" + "="*30)
    # print("      TOWER TYCOON      ")
    # print("="*30)
    # print(env.game_description)
    # print("\n" + env.user_guide)
    # print("="*30 + "\n")
    #
    # while running:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP: movement = 1
    #             elif event.key == pygame.K_DOWN: movement = 2
    #             elif event.key == pygame.K_LEFT: movement = 3
    #             elif event.key == pygame.K_RIGHT: movement = 4
    #             elif event.key == pygame.K_SPACE: space = 1
    #             elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
    #
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated or truncated:
    #         print(f"Game Over! Final Balance: ${info['score']:.2f}, Steps: {info['steps']}")
    #         pygame.time.wait(3000)
    #         obs, info = env.reset()
    #
    #     clock.tick(10) # Run at 10 steps per second for manual play
    # pygame.quit()


    # --- Random Agent Example ---
    env = GameEnv()
    obs, info = env.reset(seed=42)
    total_reward = 0
    for i in range(1500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}: Balance: ${info['score']:.2f}, Floors: {info['floors']}, Total Reward: {total_reward:.2f}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Balance: ${info['score']:.2f}")
            obs, info = env.reset(seed=42)
            total_reward = 0
            break
    env.close()