import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:29:18.185518
# Source Brief: brief_02417.md
# Brief Index: 2417
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for Robotic Arm Assembly.

    The agent controls three robotic arms to assemble five widgets within a time limit.
    This environment prioritizes visual quality and engaging gameplay.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Press to cycle active arm
    - actions[2]: Shift button (0=released, 1=held) -> Press to grab/release component

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +2.1 for correctly placing a component and completing a widget.
    - -0.1 for placing a component in the wrong station.
    - -0.001 per step to encourage speed.
    - +50 for completing all widgets.
    - -20 for running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control robotic arms to pick up components and assemble them into widgets at designated stations before time runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the active arm. Press space to cycle between arms and shift to grab or release a component."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed for visual smoothness, not step rate
        self.MAX_STEPS = 1800  # 60 seconds * 30 FPS
        self.ARM_SPEED = 4.0
        self.NUM_WIDGETS = 5
        self.NUM_ARMS = 3
        self.GRAB_RADIUS = 20

        # --- Colors (Futuristic/Industrial Theme) ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_STATION = (40, 50, 60)
        self.COLOR_STATION_TARGET = (50, 65, 80)
        self.COLOR_ARM_INACTIVE = (120, 130, 140)
        self.COLOR_ARM_ACTIVE = (0, 200, 255)
        self.COLOR_ARM_ACTIVE_GLOW = (0, 200, 255, 50)
        self.COLOR_SUCCESS = (0, 255, 120)
        self.COLOR_FAILURE = (255, 80, 80)
        self.COLOR_TEXT = (220, 230, 240)
        self.COMPONENT_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
            self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_small = pygame.font.SysFont("sans", 16)
            self.font_medium = pygame.font.SysFont("sans", 24)
            self.font_large = pygame.font.SysFont("sans", 48)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_reward = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.arms = []
        self.stations = []
        self.components = []
        self.particles = []
        self.selected_arm_index = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_reward = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.selected_arm_index = 0
        self.particles = []

        self._init_arms()
        self._init_widgets_and_stations()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.last_reward = 0.0

        if not self.game_over:
            self._handle_input(action)
            self._update_arms(action)
            self._update_components()
            self._update_particles()
            
            reward -= 0.001  # Small time penalty

        self.score += self.last_reward
        reward += self.last_reward

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if all(s['completed'] for s in self.stations):
                # sound: "Game_Win"
                reward += 50.0
                self.score += 50.0
            else:
                # sound: "Game_Lose"
                reward += -20.0
                self.score += -20.0

        truncated = False # This environment does not truncate
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _init_arms(self):
        self.arms = []
        base_positions = [
            pygame.Vector2(30, self.HEIGHT // 4),
            pygame.Vector2(30, self.HEIGHT // 2),
            pygame.Vector2(30, self.HEIGHT * 3 // 4),
        ]
        speed_multipliers = [0.9, 1.0, 1.1]
        random.shuffle(speed_multipliers)

        for i in range(self.NUM_ARMS):
            self.arms.append({
                'base_pos': base_positions[i],
                'pos': pygame.Vector2(base_positions[i]),
                'speed_mult': speed_multipliers[i],
                'held_component_id': None,
            })

    def _init_widgets_and_stations(self):
        self.stations = []
        self.components = []
        station_y = self.HEIGHT // 2
        station_spacing = (self.WIDTH - 200) / (self.NUM_WIDGETS - 1)

        for i in range(self.NUM_WIDGETS):
            station_x = 150 + i * station_spacing
            station_pos = pygame.Vector2(station_x, station_y)
            self.stations.append({
                'pos': station_pos,
                'rect': pygame.Rect(station_x - 25, station_y - 25, 50, 50),
                'progress': 0.0,
                'completed': False,
            })
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            offset = self.np_random.uniform(60, 100)
            comp_pos = pygame.Vector2(
                station_pos.x + math.cos(angle) * offset,
                station_pos.y + math.sin(angle) * offset
            )
            self.components.append({
                'id': i,
                'pos': comp_pos,
                'color': self.COMPONENT_COLORS[i],
                'is_held': False,
            })

    def _handle_input(self, action):
        space_held, shift_held = action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if space_pressed:
            # sound: "UI_Switch_01"
            self.selected_arm_index = (self.selected_arm_index + 1) % self.NUM_ARMS

        if shift_pressed:
            self._handle_grab_release()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _handle_grab_release(self):
        active_arm = self.arms[self.selected_arm_index]

        # --- Try to RELEASE a component ---
        if active_arm['held_component_id'] is not None:
            # sound: "Arm_Release_01"
            component_id = active_arm['held_component_id']
            component = self.components[component_id]
            component['is_held'] = False
            active_arm['held_component_id'] = None

            for i, station in enumerate(self.stations):
                if station['rect'].collidepoint(component['pos']):
                    if i == component_id and not station['completed']: # Correct station
                        # sound: "Placement_Success_01"
                        station['completed'] = True
                        component['pos'] = pygame.Vector2(station['pos'])
                        self.last_reward += 2.1  # +2 widget, +0.1 placement
                        self._create_particles(component['pos'], self.COLOR_SUCCESS, 50, 2)
                        return
                    else: # Wrong station
                        # sound: "Placement_Fail_01"
                        self.last_reward -= 0.1
                        self._create_particles(component['pos'], self.COLOR_FAILURE, 20, 1)
                        return
        
        # --- Try to GRAB a component ---
        else:
            for component in self.components:
                if not component['is_held'] and not self.stations[component['id']]['completed']:
                    dist = active_arm['pos'].distance_to(component['pos'])
                    if dist < self.GRAB_RADIUS:
                        # sound: "Arm_Grab_01"
                        active_arm['held_component_id'] = component['id']
                        component['is_held'] = True
                        break

    def _update_arms(self, action):
        movement = action[0]
        active_arm = self.arms[self.selected_arm_index]
        speed = self.ARM_SPEED * active_arm['speed_mult']

        vel = pygame.Vector2(0, 0)
        if movement == 1: vel.y = -1
        elif movement == 2: vel.y = 1
        elif movement == 3: vel.x = -1
        elif movement == 4: vel.x = 1

        if vel.length() > 0:
            vel.normalize_ip()
            active_arm['pos'] += vel * speed

        active_arm['pos'].x = max(10, min(self.WIDTH - 10, active_arm['pos'].x))
        active_arm['pos'].y = max(10, min(self.HEIGHT - 10, active_arm['pos'].y))

    def _update_components(self):
        for component in self.components:
            if component['is_held']:
                for arm in self.arms:
                    if arm['held_component_id'] == component['id']:
                        component['pos'] = pygame.Vector2(arm['pos'])
                        break

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if all(s['completed'] for s in self.stations):
            return True
        return False

    def _create_particles(self, pos, color, count, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.0) * speed
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_mag
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'color': color,
                'life': self.np_random.integers(15, 30),
            })

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        # Render stations
        for station in self.stations:
            pygame.draw.rect(self.screen, self.COLOR_STATION, station['rect'], border_radius=5)
            if not station['completed']:
                pygame.draw.rect(self.screen, self.COLOR_STATION_TARGET, station['rect'].inflate(-8, -8), 2, border_radius=3)
            else:
                 pygame.gfxdraw.filled_circle(self.screen, int(station['pos'].x), int(station['pos'].y), 15, self.components[self.stations.index(station)]['color'])


        # Render components
        for component in self.components:
            if not component['is_held'] and not self.stations[component['id']]['completed']:
                pygame.gfxdraw.filled_circle(self.screen, int(component['pos'].x), int(component['pos'].y), 8, component['color'])
                pygame.gfxdraw.aacircle(self.screen, int(component['pos'].x), int(component['pos'].y), 8, component['color'])

        # Render arms
        for i, arm in enumerate(self.arms):
            is_active = (i == self.selected_arm_index)
            color = self.COLOR_ARM_ACTIVE if is_active else self.COLOR_ARM_INACTIVE

            # Arm track
            pygame.draw.line(self.screen, self.COLOR_STATION, (arm['base_pos'].x, 0), (arm['base_pos'].x, self.HEIGHT), 4)
            # Arm base
            pygame.draw.rect(self.screen, color, (arm['base_pos'].x - 10, arm['pos'].y - 10, 20, 20), border_radius=3)
            # Arm beam
            pygame.draw.line(self.screen, color, (arm['base_pos'].x, arm['pos'].y), (arm['pos'].x, arm['pos'].y), 3)
            
            # Effector (end of arm)
            pos_int = (int(arm['pos'].x), int(arm['pos'].y))
            if is_active:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 12, self.COLOR_ARM_ACTIVE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, self.COLOR_BG)

            if arm['held_component_id'] is not None:
                comp_color = self.components[arm['held_component_id']]['color']
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 6, comp_color)

        # Render particles
        for p in self.particles:
            size = max(1, int(p['life'] / 10))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_FAILURE
        timer_text = self.font_medium.render(f"Time: {time_left:.1f}s", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 5))

        # Arm speed info
        for arm in self.arms:
            speed_text = self.font_small.render(f"{int(arm['speed_mult']*100)}%", True, self.COLOR_TEXT)
            self.screen.blit(speed_text, (arm['base_pos'].x - 12, arm['base_pos'].y - 25))
            
        # Progress bars over stations
        for i, station in enumerate(self.stations):
            bar_rect = pygame.Rect(station['rect'].left, station['rect'].top - 12, station['rect'].width, 8)
            pygame.draw.rect(self.screen, self.COLOR_STATION, bar_rect, border_radius=2)
            if station['completed']:
                fill_rect = pygame.Rect(bar_rect.left, bar_rect.top, bar_rect.width, bar_rect.height)
                pygame.draw.rect(self.screen, self.COMPONENT_COLORS[i], fill_rect, border_radius=2)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(s['completed'] for s in self.stations)
            msg = "ASSEMBLY COMPLETE" if win_condition else "TIME'S UP"
            color = self.COLOR_SUCCESS if win_condition else self.COLOR_FAILURE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you will need to `pip install pygame`
    # It is not included in the environment's dependencies.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Robotic Arm Assembly - Manual Test")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move active arm")
    print("Space: Cycle active arm")
    print("Shift: Grab/Release component")
    print("Q: Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(env.FPS)
        
    pygame.quit()