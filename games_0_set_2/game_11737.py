import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:31:31.083534
# Source Brief: brief_01737.md
# Brief Index: 1737
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-level Gymnasium environment for a strategy/resource management game.
    The player must build an ark to survive a rising flood by gathering resources
    from different settlements, switching forms, and managing time.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Build an ark to survive a rising flood. Gather resources from settlements "
        "and manage your time before the water overwhelms you."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport between settlements. "
        "Press space to build the ark and shift to switch between land/water forms."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.FPS = 30

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24)

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_WATER = (60, 120, 220)
        self.COLOR_WATER_WAVE = (150, 180, 240)
        self.COLOR_ARK = (255, 200, 0)
        self.COLOR_ARK_SHADOW = (180, 140, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PLAYER_LAND = (50, 255, 50)
        self.COLOR_PLAYER_WATER = (255, 50, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 200)
        self.COLOR_SETTLEMENT_LAND = (139, 69, 19)
        self.COLOR_SETTLEMENT_WATER = (0, 150, 150)
        self.COLOR_SETTLEMENT_INACTIVE = (80, 80, 80)
        
        # Game Mechanic Constants
        self.ARK_INITIAL_HEIGHT = 40.0
        self.ARK_BUILD_AMOUNT = 6.0
        self.ARK_BUILD_COST = {'wood': 10, 'fish': 5}
        self.WATER_INITIAL_LEVEL = 20.0
        self.WATER_BASE_RISE_RATE = 0.15
        self.WATER_DIFFICULTY_SCALING_RATE = 0.05
        self.WATER_DIFFICULTY_SCALING_INTERVAL = 200
        self.INACTIVE_SETTLEMENT_PENALTY = 0.04
        self.SETTLEMENT_INITIAL_RESOURCES = 100
        self.SETTLEMENT_COOLDOWN = 50
        self.RESOURCE_GATHER_AMOUNT = 5

        # State variables are initialized in reset()
        self.settlements = []
        self.player_pos_index = 0
        self.player_form = 'land'
        self.resources = {}
        self.ark_height = 0.0
        self.water_level = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        self._setup_settlements()
        # self.reset() is called by the environment wrapper

    def _setup_settlements(self):
        w, h = self.WIDTH, self.HEIGHT
        self.settlement_positions = [
            (w * 0.5, h * 0.25), # 0: Top (Land)
            (w * 0.5, h * 0.75), # 1: Bottom (Land)
            (w * 0.2, h * 0.5),  # 2: Left (Water)
            (w * 0.8, h * 0.5),  # 3: Right (Water)
        ]
        self.settlement_types = ['land', 'land', 'water', 'water']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.ark_height = self.ARK_INITIAL_HEIGHT
        self.water_level = self.WATER_INITIAL_LEVEL
        
        self.resources = {'wood': 20, 'fish': 20}
        self.player_pos_index = 0
        self.player_form = 'land'
        
        self.last_space_held = False
        self.last_shift_held = False

        self.particles = []

        self.settlements = []
        for i in range(4):
            self.settlements.append({
                'pos': self.settlement_positions[i],
                'type': self.settlement_types[i],
                'resources': self.SETTLEMENT_INITIAL_RESOURCES,
                'cooldown': 0
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle Player Actions (rising edge detection) ---
        
        # Action: Switch Form (Shift)
        if shift_action and not self.last_shift_held:
            self.player_form = 'water' if self.player_form == 'land' else 'land'
            self._spawn_particles(self.settlement_positions[self.player_pos_index], 15, self.COLOR_PLAYER_GLOW, 'burst')
            # sfx: form_switch_sound

        # Action: Build Ark (Space)
        if space_action and not self.last_space_held:
            if self.resources['wood'] >= self.ARK_BUILD_COST['wood'] and self.resources['fish'] >= self.ARK_BUILD_COST['fish']:
                self.resources['wood'] -= self.ARK_BUILD_COST['wood']
                self.resources['fish'] -= self.ARK_BUILD_COST['fish']
                self.ark_height += self.ARK_BUILD_AMOUNT
                reward += 0.5
                self._spawn_particles((self.WIDTH / 2, self.HEIGHT - self.ark_height), 20, self.COLOR_ARK, 'fountain')
                # sfx: build_sound

        # Action: Teleport (Movement)
        if movement in [1, 2, 3, 4]:
            new_pos_index = movement - 1
            if self.player_pos_index != new_pos_index:
                old_pos = self.settlement_positions[self.player_pos_index]
                self.player_pos_index = new_pos_index
                new_pos = self.settlement_positions[self.player_pos_index]
                self._spawn_particles(old_pos, 20, (255, 255, 255), 'implode')
                self._spawn_particles(new_pos, 20, (255, 255, 255), 'burst')
                # sfx: teleport_sound

        self.last_space_held = space_action
        self.last_shift_held = shift_action

        # --- Update Game State ---
        
        # Automatic: Gather Resources
        current_settlement = self.settlements[self.player_pos_index]
        if current_settlement['cooldown'] == 0:
            resource_type = 'wood' if current_settlement['type'] == 'land' else 'fish'
            player_resource_type = 'wood' if self.player_form == 'land' else 'fish'
            
            if resource_type == player_resource_type:
                gathered = min(self.RESOURCE_GATHER_AMOUNT, current_settlement['resources'])
                if gathered > 0:
                    self.resources[resource_type] += gathered
                    current_settlement['resources'] -= gathered
                    reward += 0.1
                    gather_color = self.COLOR_SETTLEMENT_LAND if resource_type == 'wood' else self.COLOR_SETTLEMENT_WATER
                    self._spawn_particles(current_settlement['pos'], 5, gather_color, 'gather')
                    # sfx: gather_sound
                
                if current_settlement['resources'] <= 0:
                    current_settlement['cooldown'] = self.SETTLEMENT_COOLDOWN

        # Update settlement cooldowns
        num_inactive_settlements = 0
        for settlement in self.settlements:
            if settlement['cooldown'] > 0:
                settlement['cooldown'] -= 1
                if settlement['cooldown'] == 0:
                    settlement['resources'] = self.SETTLEMENT_INITIAL_RESOURCES
                num_inactive_settlements += 1

        # Update water level
        difficulty_bonus = self.WATER_DIFFICULTY_SCALING_RATE * (self.steps // self.WATER_DIFFICULTY_SCALING_INTERVAL)
        inactive_penalty = self.INACTIVE_SETTLEMENT_PENALTY * num_inactive_settlements
        self.water_level += self.WATER_BASE_RISE_RATE + difficulty_bonus + inactive_penalty

        self.steps += 1
        self.score += reward

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.water_level >= self.ark_height:
            terminated = True
            reward -= 100
            self.score -= 100
            self.game_over = True
            # sfx: game_over_drown_sound
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            reward += 100
            self.score += 100
            self.game_over = True
            # sfx: victory_sound

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "ark_height": self.ark_height, "water_level": self.water_level}

    def _render_game(self):
        self._update_and_draw_particles()
        self._draw_water()
        self._draw_ark()
        self._draw_settlements()
        self._draw_player()

    def _draw_water(self):
        water_y = self.HEIGHT - self.water_level
        water_rect = pygame.Rect(0, water_y, self.WIDTH, self.HEIGHT - water_y)
        pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect)
        
        # Wave effect
        for i in range(self.WIDTH):
            wave_offset = math.sin((i / 20.0) + (self.steps / 10.0)) * 2
            pygame.draw.line(self.screen, self.COLOR_WATER_WAVE, (i, water_y + wave_offset), (i, water_y + wave_offset), 2)

    def _draw_ark(self):
        ark_base_y = self.HEIGHT
        ark_top_y = self.HEIGHT - self.ark_height
        ark_width = 60
        ark_x = self.WIDTH / 2 - ark_width / 2

        # Draw main structure
        ark_rect = pygame.Rect(ark_x, ark_top_y, ark_width, self.ark_height)
        pygame.draw.rect(self.screen, self.COLOR_ARK, ark_rect)
        pygame.draw.rect(self.screen, self.COLOR_ARK_SHADOW, ark_rect, 3)

        # Draw compartments
        for y in range(int(ark_base_y), int(ark_top_y), -10):
            pygame.draw.line(self.screen, self.COLOR_ARK_SHADOW, (ark_x, y), (ark_x + ark_width, y), 1)

    def _draw_settlements(self):
        for i, s in enumerate(self.settlements):
            pos = (int(s['pos'][0]), int(s['pos'][1]))
            radius = 15
            
            if s['cooldown'] > 0:
                color = self.COLOR_SETTLEMENT_INACTIVE
            elif s['type'] == 'land':
                color = self.COLOR_SETTLEMENT_LAND
            else:
                color = self.COLOR_SETTLEMENT_WATER
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, tuple(c//2 for c in color))

            # Highlight if player is here
            if i == self.player_pos_index:
                pulse = abs(math.sin(self.steps * 0.2))
                highlight_color = (*self.COLOR_PLAYER_GLOW[:3], int(100 * pulse))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, highlight_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 5, highlight_color)


    def _draw_player(self):
        pos = self.settlement_positions[self.player_pos_index]
        x, y = int(pos[0]), int(pos[1])
        size = 10
        
        # Draw glow
        for i in range(5, 0, -1):
            alpha = 50 - i * 10
            pygame.gfxdraw.filled_circle(self.screen, x, y, size + i * 2, (*self.COLOR_PLAYER_GLOW[:3], alpha))

        if self.player_form == 'land':
            color = self.COLOR_PLAYER_LAND
            points = [
                (x, y - size),
                (x - size, y + size // 2),
                (x + size, y + size // 2)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else: # water form
            color = self.COLOR_PLAYER_WATER
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)

    def _render_ui(self):
        # Top Left: Time
        time_text = self.font_small.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Top Right: Resources
        wood_text = self.font_small.render(f"Wood: {self.resources['wood']}", True, self.COLOR_TEXT)
        fish_text = self.font_small.render(f"Fish: {self.resources['fish']}", True, self.COLOR_TEXT)
        self.screen.blit(wood_text, (self.WIDTH - wood_text.get_width() - 10, 10))
        self.screen.blit(fish_text, (self.WIDTH - fish_text.get_width() - 10, 30))
        
        # Top Center: Levels
        ark_h = int(self.ark_height)
        water_l = int(self.water_level)
        level_text = self.font_small.render(f"Ark: {ark_h}m | Water: {water_l}m", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH/2 - level_text.get_width()/2, 10))

        # Bottom Center: Player Form
        form_text_str = f"Form: {self.player_form.upper()}"
        form_color = self.COLOR_PLAYER_LAND if self.player_form == 'land' else self.COLOR_PLAYER_WATER
        form_text = self.font_small.render(form_text_str, True, form_color)
        self.screen.blit(form_text, (self.WIDTH/2 - form_text.get_width()/2, self.HEIGHT - 30))

        # Game Over/Win Text
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                msg = "VICTORY! The Flood Receded."
                color = (150, 255, 150)
            else:
                msg = "GAME OVER. The Ark Was Flooded."
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'burst':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            elif p_type == 'fountain':
                vel = [random.uniform(-1, 1), random.uniform(-5, -1)]
            elif p_type == 'implode':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = [-math.cos(angle) * speed, -math.sin(angle) * speed]
            elif p_type == 'gather':
                vel = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
            else:
                vel = [0,0]

            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
                continue

            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(255 * (p['life'] / 40.0))
                color_with_alpha = (*p['color'][:3], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color_with_alpha)

    def close(self):
        pygame.quit()

    def render(self):
        # This method is not strictly required for rgb_array mode but is good practice.
        return self._get_observation()

if __name__ == '__main__':
    # Example usage: Play the game with random actions or manual control
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ark Survival Environment")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Use a dictionary to track key presses for manual play
    keys_pressed = {
        'up': False, 'down': False, 'left': False, 'right': False,
        'space': False, 'shift': False
    }

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    print("R: Reset")
    print("A: Toggle Manual/Auto mode")
    print("----------------------\n")

    manual_mode = True
    running = True

    while running:
        # --- Event Handling for Manual Play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                if event.key == pygame.K_UP: keys_pressed['up'] = True
                if event.key == pygame.K_DOWN: keys_pressed['down'] = True
                if event.key == pygame.K_LEFT: keys_pressed['left'] = True
                if event.key == pygame.K_RIGHT: keys_pressed['right'] = True
                if event.key == pygame.K_SPACE: keys_pressed['space'] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_pressed['shift'] = True
                if event.key == pygame.K_a:
                    manual_mode = not manual_mode
                    print(f"Switched to {'Manual' if manual_mode else 'Automatic (Random Agent)'} mode.")

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_pressed['up'] = False
                if event.key == pygame.K_DOWN: keys_pressed['down'] = False
                if event.key == pygame.K_LEFT: keys_pressed['left'] = False
                if event.key == pygame.K_RIGHT: keys_pressed['right'] = False
                if event.key == pygame.K_SPACE: keys_pressed['space'] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_pressed['shift'] = False

        if not done:
            action_to_take = None
            if manual_mode:
                # Construct action from key presses
                move_action = 0
                if keys_pressed['up']: move_action = 1
                elif keys_pressed['down']: move_action = 2
                elif keys_pressed['left']: move_action = 3
                elif keys_pressed['right']: move_action = 4
                
                space_action = 1 if keys_pressed['space'] else 0
                shift_action = 1 if keys_pressed['shift'] else 0
                
                action_to_take = [move_action, space_action, shift_action]
            else:
                # Automatic random agent
                action_to_take = env.action_space.sample()
            
            # For auto-advance games, we always step
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()