import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:34:13.776519
# Source Brief: brief_03185.md
# Brief Index: 3185
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
        "Evade security cameras and otherworldly entities in a derelict facility. Scavenge for scrap to craft tools and repair machinery to survive until dawn."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to interact with objects or open the crafting menu. Press shift to use crafted items."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 1000
    PLAYER_SPEED = 3.0
    PLAYER_SIZE = 12
    CAMERA_DETECTION_LIMIT = 50

    # --- Colors (VHS Style) ---
    COLOR_BG = (10, 15, 30)
    COLOR_WALL = (40, 50, 80)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_PLAYER_GLOW = (255, 255, 100, 30)
    COLOR_CAMERA = (180, 180, 200)
    COLOR_CAMERA_CONE = (255, 0, 0)
    COLOR_ENTITY = (255, 80, 120)
    COLOR_RESOURCE = (50, 255, 50)
    COLOR_MACHINERY_OFF = (50, 100, 255)
    COLOR_MACHINERY_ON = (80, 80, 100)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_UI_BG = (20, 30, 60, 180)
    COLOR_CRAFT_SELECT = (255, 255, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont('Consolas', 20, bold=True)
            self.font_small = pygame.font.SysFont('Consolas', 16)
            self.font_title = pygame.font.SysFont('Consolas', 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 24)
            self.font_small = pygame.font.SysFont(None, 20)
            self.font_title = pygame.font.SysFont(None, 60)

        # VHS effect surfaces
        self._vhs_static_surface = self._create_static_surface()
        self._vhs_scanlines_surface = self._create_scanlines_surface()
        
        # Initialize state variables
        self.player = {}
        self.cameras = []
        self.entities = []
        self.resources = []
        self.machinery = []
        self.repel_effect = None
        
        # Initialize state variables by calling reset
        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # Player state
        self.player = {
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32),
            "vel": np.array([0, 0], dtype=np.float32),
            "detection": 0.0,
            "scrap": 0,
            "emitters": 0,
        }

        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False
        self.crafting_menu_open = False
        self.crafting_selection = 0

        # World state
        self.cameras = self._initialize_cameras()
        self.resources = self._initialize_resources()
        self.machinery = self._initialize_machinery()
        self.entities = []
        self.entity_spawn_cooldown = 0
        self.repel_effect = None

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # 1. Handle Input
            reward += self._handle_input(action)
            
            # 2. Update Game State
            self._update_player()
            self._update_cameras()
            self._update_entities()
            self._update_resources()
            self._update_repel_effect()
            self._update_progression()

            # 3. Calculate Rewards & Check Termination
            step_reward, terminated = self._calculate_rewards_and_termination()
            reward += step_reward
            if terminated:
                self.game_over = True
                if self.steps >= self.MAX_STEPS:
                    self.game_won = True
                    reward += 100
                else:
                    reward -= 100

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        reward = 0

        # Movement Velocity
        vel = np.array([0, 0], dtype=np.float32)
        if movement == 1: vel[1] = -1
        elif movement == 2: vel[1] = 1
        elif movement == 3: vel[0] = -1
        elif movement == 4: vel[0] = 1
        
        if np.linalg.norm(vel) > 0:
            vel = vel / np.linalg.norm(vel)
        self.player['vel'] = vel * self.PLAYER_SPEED

        # Crafting Menu Logic
        if self.crafting_menu_open:
            if space_pressed:
                crafted, craft_reward = self._craft_item()
                if crafted:
                    reward += craft_reward
                self.crafting_menu_open = False
            elif movement in [1, 2] and (self.steps % 5 == 0): # Add a small delay to selection
                self.crafting_selection = (self.crafting_selection + (1 if movement == 2 else -1)) % 2
        else: # Menu is closed
            if space_pressed:
                # Check for machinery interaction first
                interacted, repair_reward = self._interact_machinery()
                if interacted:
                    reward += repair_reward
                else:
                    self.crafting_menu_open = True

        # Use Item Logic
        if shift_pressed and not self.crafting_menu_open:
            if self.player['emitters'] > 0 and self.repel_effect is None:
                self.player['emitters'] -= 1
                self.repel_effect = {"pos": self.player['pos'].copy(), "radius": 10, "duration": 90}
                # SFX: Sonic Emitter Activation

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _update_player(self):
        self.player['pos'] += self.player['vel']
        self.player['pos'][0] = np.clip(self.player['pos'][0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player['pos'][1] = np.clip(self.player['pos'][1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _update_cameras(self):
        is_detected = False
        for cam in self.cameras:
            cam['angle'] = (cam['angle'] + cam['speed']) % 360
            if self._is_point_in_cone(self.player['pos'], cam):
                is_detected = True
                break
        
        if is_detected:
            self.player['detection'] = min(self.CAMERA_DETECTION_LIMIT, self.player['detection'] + 1)
        else:
            self.player['detection'] = max(0, self.player['detection'] - 0.5)

    def _update_entities(self):
        for entity in self.entities[:]:
            # Repel logic
            if self.repel_effect and np.linalg.norm(entity['pos'] - self.repel_effect['pos']) < self.repel_effect['radius']:
                self.entities.remove(entity)
                # SFX: Entity banished
                continue

            # Movement logic
            direction = self.player['pos'] - entity['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            entity['pos'] += direction * entity['speed']
            
            # Flicker effect
            entity['flicker'] = random.randint(0, 5)

    def _update_resources(self):
        for res in self.resources:
            if res['collected']:
                res['timer'] -= 1
                if res['timer'] <= 0:
                    res['collected'] = False
                    # SFX: Resource respawn

    def _update_repel_effect(self):
        if self.repel_effect:
            self.repel_effect['duration'] -= 1
            self.repel_effect['radius'] += 2
            if self.repel_effect['duration'] <= 0:
                self.repel_effect = None
    
    def _update_progression(self):
        self.entity_spawn_cooldown = max(0, self.entity_spawn_cooldown - 1)
        
        spawn_rate = 150
        num_repaired = sum(1 for m in self.machinery if m['repaired'])
        spawn_rate += num_repaired * 50 # Each repaired machine slows spawn

        if self.steps > 200 and self.entity_spawn_cooldown == 0:
            self.entity_spawn_cooldown = random.randint(int(spawn_rate * 0.8), int(spawn_rate * 1.2))
            self._spawn_entity()

    def _calculate_rewards_and_termination(self):
        reward = 0
        terminated = False

        # Resource collection
        for res in self.resources:
            if not res['collected'] and np.linalg.norm(self.player['pos'] - res['pos']) < self.PLAYER_SIZE + 5:
                res['collected'] = True
                res['timer'] = 300 # Respawn time
                self.player['scrap'] += 1
                reward += 1
                # SFX: Collect scrap

        # Camera detection penalty
        if self.player['detection'] > 0:
            reward -= 0.1
        
        # Termination conditions
        if self.player['detection'] >= self.CAMERA_DETECTION_LIMIT:
            terminated = True
        
        for entity in self.entities:
            if np.linalg.norm(self.player['pos'] - entity['pos']) < self.PLAYER_SIZE:
                reward -= 5
                terminated = True
                break
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return reward, terminated
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_world()
        self._render_vhs_overlay()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_world(self):
        # Machinery
        for machine in self.machinery:
            color = self.COLOR_MACHINERY_ON if machine['repaired'] else self.COLOR_MACHINERY_OFF
            pygame.draw.rect(self.screen, color, machine['rect'])
            pygame.draw.rect(self.screen, self.COLOR_WALL, machine['rect'], 2)

        # Resources
        for res in self.resources:
            if not res['collected']:
                self._draw_glowing_circle(self.screen, self.COLOR_RESOURCE, res['pos'], 8, 15)

        # Camera cones and bodies
        for cam in self.cameras:
            self._render_camera_cone(cam)
            pygame.draw.circle(self.screen, self.COLOR_CAMERA, cam['pos'], 8)

        # Repel Effect
        if self.repel_effect:
            radius = int(self.repel_effect['radius'])
            alpha = int(100 * (self.repel_effect['duration'] / 90.0))
            if radius > 0 and alpha > 0:
                self._draw_glowing_circle(self.screen, (200, 255, 255, alpha), self.repel_effect['pos'], radius, 10)

        # Player
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.player['pos'], self.PLAYER_SIZE, 20)

        # Entities
        for entity in self.entities:
            if entity['flicker'] > 0:
                size = 10 + entity['flicker']
                self._draw_glowing_circle(self.screen, self.COLOR_ENTITY, entity['pos'], size, 15)

    def _render_vhs_overlay(self):
        # Chromatic aberration
        self.screen.blit(self.screen, (2, 0), special_flags=pygame.BLEND_RGB_ADD)
        self.screen.blit(self.screen, (-2, 0), special_flags=pygame.BLEND_RGB_SUB)

        # Static and scanlines
        self.screen.blit(self._vhs_static_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(self._vhs_scanlines_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    def _render_ui(self):
        # Top-left info panel
        ui_panel = pygame.Surface((200, 80), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self._draw_text(ui_panel, f"SCRAP:    {self.player['scrap']}", (10, 10), self.font_small, self.COLOR_UI_TEXT)
        self._draw_text(ui_panel, f"EMITTERS: {self.player['emitters']}", (10, 30), self.font_small, self.COLOR_UI_TEXT)
        dawn_time = (self.steps / self.MAX_STEPS) * 6
        self._draw_text(ui_panel, f"TIME: {int(dawn_time):02d}:00 AM", (10, 50), self.font_small, self.COLOR_UI_TEXT)
        self.screen.blit(ui_panel, (10, 10))

        # Detection Meter
        if self.player['detection'] > 0:
            bar_w = 200
            bar_h = 20
            fill_w = (self.player['detection'] / self.CAMERA_DETECTION_LIMIT) * bar_w
            
            x = (self.SCREEN_WIDTH - bar_w) / 2
            y = self.SCREEN_HEIGHT - 40
            
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, (x, y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_CAMERA_CONE, (x, y, fill_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x, y, bar_w, bar_h), 1)
            self._draw_text(self.screen, "!! DETECTED !!", (x + bar_w/2, y + bar_h/2), self.font_small, self.COLOR_UI_TEXT, center=True)

        # Crafting Menu
        if self.crafting_menu_open:
            self._render_crafting_menu()

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "DAWN      SURVIVED" if self.game_won else "CONNECTION LOST"
            self._draw_text(self.screen, msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.font_title, self.COLOR_PLAYER, center=True)
            
    def _render_crafting_menu(self):
        w, h = 300, 150
        x, y = (self.SCREEN_WIDTH - w) / 2, (self.SCREEN_HEIGHT - h) / 2
        
        menu_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        menu_surf.fill(self.COLOR_UI_BG)
        pygame.draw.rect(menu_surf, self.COLOR_UI_TEXT, (0, 0, w, h), 2)
        
        self._draw_text(menu_surf, "[ CRAFTING ]", (w/2, 20), self.font_main, self.COLOR_UI_TEXT, center=True)
        
        # Item 1: Sonic Emitter
        cost1 = 3
        color1 = self.COLOR_UI_TEXT if self.player['scrap'] >= cost1 else self.COLOR_CAMERA_CONE
        self._draw_text(menu_surf, f"Sonic Emitter (Cost: {cost1} Scrap)", (20, 60), self.font_small, color1)
        
        # Item 2: Repair Machinery
        cost2 = 5
        color2 = self.COLOR_UI_TEXT if self.player['scrap'] >= cost2 else self.COLOR_CAMERA_CONE
        self._draw_text(menu_surf, f"Repair Machinery (Cost: {cost2} Scrap)", (20, 90), self.font_small, color2)

        # Selector
        sel_y = 60 + self.crafting_selection * 30
        pygame.draw.rect(menu_surf, self.COLOR_CRAFT_SELECT, (10, sel_y - 2, w - 20, 20), 1)
        
        self.screen.blit(menu_surf, (x, y))

    def _render_camera_cone(self, cam):
        num_points = 20
        points = [cam['pos']]
        angle_start = cam['angle'] - cam['fov'] / 2
        
        for i in range(num_points + 1):
            angle = math.radians(angle_start + (i / num_points) * cam['fov'])
            x = cam['pos'][0] + cam['range'] * math.cos(angle)
            y = cam['pos'][1] + cam['range'] * math.sin(angle)
            points.append((x, y))

        alpha = 100 + 30 * math.sin(self.steps * 0.1)
        color = (*self.COLOR_CAMERA_CONE, int(alpha))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "scrap": self.player['scrap']}

    # --- Helper Methods ---
    def _initialize_cameras(self):
        return [
            {'pos': np.array([50, 50]), 'angle': 45, 'speed': 0.5, 'fov': 60, 'range': 150},
            {'pos': np.array([self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT - 50]), 'angle': 225, 'speed': -0.7, 'fov': 70, 'range': 200},
            {'pos': np.array([self.SCREEN_WIDTH - 80, 80]), 'angle': 135, 'speed': 0.3, 'fov': 50, 'range': 180},
        ]

    def _initialize_resources(self):
        return [
            {'pos': np.array([100, 300]), 'collected': False, 'timer': 0},
            {'pos': np.array([540, 100]), 'collected': False, 'timer': 0},
            {'pos': np.array([320, 50]), 'collected': False, 'timer': 0},
            {'pos': np.array([320, 350]), 'collected': False, 'timer': 0},
        ]

    def _initialize_machinery(self):
        return [
            {'rect': pygame.Rect(20, self.SCREEN_HEIGHT/2 - 25, 30, 50), 'repaired': False},
            {'rect': pygame.Rect(self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT/2 - 25, 30, 50), 'repaired': False},
        ]

    def _spawn_entity(self):
        speed = 0.5 + (self.steps / self.MAX_STEPS) * 1.5
        
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': pos = np.array([random.uniform(0, self.SCREEN_WIDTH), -10])
        elif edge == 'bottom': pos = np.array([random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10])
        elif edge == 'left': pos = np.array([-10, random.uniform(0, self.SCREEN_HEIGHT)])
        else: pos = np.array([self.SCREEN_WIDTH + 10, random.uniform(0, self.SCREEN_HEIGHT)])
        
        self.entities.append({'pos': pos, 'speed': speed, 'flicker': 0})
        # SFX: Entity spawn whisper

    def _is_point_in_cone(self, point, cam):
        vec_to_point = point - cam['pos']
        dist = np.linalg.norm(vec_to_point)
        if dist == 0 or dist > cam['range']:
            return False
        
        angle_to_point = math.degrees(math.atan2(vec_to_point[1], vec_to_point[0]))
        cam_dir_angle = cam['angle']
        
        diff = (angle_to_point - cam_dir_angle + 180) % 360 - 180
        return abs(diff) < cam['fov'] / 2

    def _craft_item(self):
        # Craft Emitter
        if self.crafting_selection == 0:
            cost = 3
            if self.player['scrap'] >= cost:
                self.player['scrap'] -= cost
                self.player['emitters'] += 1
                # SFX: Craft success
                return True, 2
        # Crafting for repair is handled by _interact_machinery
        return False, 0
    
    def _interact_machinery(self):
        cost = 5
        if self.player['scrap'] < cost:
            return False, 0

        for machine in self.machinery:
            if not machine['repaired']:
                if pygame.Rect(*self.player['pos'], 1, 1).colliderect(machine['rect'].inflate(20, 20)):
                    self.player['scrap'] -= cost
                    machine['repaired'] = True
                    # SFX: Machine repair
                    return True, 5
        return False, 0

    def _create_static_surface(self):
        surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for _ in range(2000):
            x, y = random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)
            alpha = random.randint(10, 40)
            surface.set_at((x, y), (255, 255, 255, alpha))
        return surface

    def _create_scanlines_surface(self):
        surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(0, self.SCREEN_HEIGHT, 4):
            pygame.draw.line(surface, (0, 0, 0, 40), (0, y), (self.SCREEN_WIDTH, y), 1)
        return surface

    def _draw_text(self, surface, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        surface.blit(text_surface, text_rect)

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_size):
        pos = (int(pos[0]), int(pos[1]))
        
        # Glow effect
        glow_color = (*color[:3], 20) if len(color) == 4 else (*color, 20)
        for i in range(glow_size, 0, -2):
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius + i, glow_color)
        
        # Main circle
        main_color = color if len(color) == 4 else (*color, 255)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, main_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, main_color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    # This block will not run in the test environment, but is useful for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("VHS Survival Horror")
    clock = pygame.time.Clock()
    
    # Game loop for human interaction
    total_reward = 0
    while not done:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward}, Steps: {info['steps']}")
    pygame.quit()