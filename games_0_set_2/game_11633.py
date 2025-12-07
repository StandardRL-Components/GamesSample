import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:20:26.315877
# Source Brief: brief_01633.md
# Brief Index: 1633
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Drone Nexus'.

    The player controls three energy-linked drones (Red, Green, Blue) to activate
    a sequence of 15 switches. The core mechanic involves spatial reasoning and
    timing, as each drone has a cooldown after moving. Activating switches in
    sequence reveals glowing pathways, guiding the player. The goal is to
    activate all switches within a 60-second time limit.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
      - This moves the currently selected drone. After a move, selection
        cycles to the next drone (R -> G -> B -> R).
    - `action[1]`: Space button (unused).
    - `action[2]`: Shift button (unused).

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +10 for each switch activated.
    - +100 for winning the game (activating all 15 switches).
    - -100 for losing (timer runs out).
    - Small continuous reward/penalty for moving drones closer/further from
      the nearest inactive switch.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three drones to activate a series of switches before time runs out. "
        "All three drones must be near a switch simultaneously to activate it."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the currently selected drone. "
        "After a move, control automatically cycles to the next drone."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60.0
    MAX_STEPS = int(GAME_DURATION_SECONDS * FPS * 2) # Generous step limit

    COLOR_BG = (10, 15, 25)
    COLOR_DRONE_R = (255, 50, 50)
    COLOR_DRONE_G = (50, 255, 50)
    COLOR_DRONE_B = (50, 100, 255)
    COLOR_SWITCH_INACTIVE = (60, 60, 70)
    COLOR_SWITCH_ACTIVE = (255, 255, 255)
    COLOR_PATHWAY = (0, 220, 220)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_WARN = (255, 150, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)

    NUM_SWITCHES = 15
    NUM_DRONES = 3

    DRONE_SPEED = 40
    DRONE_RADIUS = 8
    DRONE_COOLDOWN_FRAMES = 2 * FPS

    SWITCH_RADIUS = 12
    SWITCH_ACTIVATION_DISTANCE = 20

    PLAY_AREA_PADDING = 40

    class Particle:
        def __init__(self, pos, color):
            self.pos = pygame.Vector2(pos)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.lifespan = random.randint(15, 30)
            self.color = color
            self.radius = random.uniform(2, 4)

        def update(self):
            self.pos += self.vel
            self.lifespan -= 1
            self.vel *= 0.95 # Damping

        def draw(self, surface):
            if self.lifespan > 0:
                alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
                temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
                surface.blit(temp_surf, self.pos - pygame.Vector2(self.radius, self.radius), special_flags=pygame.BLEND_RGBA_ADD)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)

        self.drones = []
        self.switches = []
        self.particles = []
        self.activated_switch_order = []
        self.active_drone_index = 0
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS
        self.active_drone_index = 0
        self.particles.clear()
        self.activated_switch_order.clear()

        # Initialize Drones
        self.drones = [
            {'pos': pygame.Vector2(0, 0), 'color': self.COLOR_DRONE_R, 'cooldown': 0},
            {'pos': pygame.Vector2(0, 0), 'color': self.COLOR_DRONE_G, 'cooldown': 0},
            {'pos': pygame.Vector2(0, 0), 'color': self.COLOR_DRONE_B, 'cooldown': 0}
        ]
        for i in range(self.NUM_DRONES):
            self.drones[i]['pos'] = self._get_random_pos()

        # Initialize Switches
        self.switches = []
        while len(self.switches) < self.NUM_SWITCHES:
            pos = self._get_random_pos()
            # Ensure switches don't overlap too much
            if all(pos.distance_to(s['pos']) > self.SWITCH_RADIUS * 2.5 for s in self.switches):
                self.switches.append({'pos': pos, 'active': False})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # --- Update Game Logic ---
        self._update_cooldowns()
        self._update_particles()
        
        reward = 0
        dist_before = self._get_total_min_distances()

        # Unpack factorized action
        movement = action[0]
        
        # --- Handle Action ---
        selected_drone = self.drones[self.active_drone_index]
        if movement != 0 and selected_drone['cooldown'] <= 0:
            # sfx: drone_move.wav
            move_vec = pygame.Vector2(0, 0)
            if movement == 1: move_vec.y = -1 # Up
            elif movement == 2: move_vec.y = 1 # Down
            elif movement == 3: move_vec.x = -1 # Left
            elif movement == 4: move_vec.x = 1 # Right
            
            selected_drone['pos'] += move_vec * self.DRONE_SPEED
            self._clamp_drone_position(selected_drone)
            selected_drone['cooldown'] = self.DRONE_COOLDOWN_FRAMES
            
            # Cycle to next drone
            self.active_drone_index = (self.active_drone_index + 1) % self.NUM_DRONES

        dist_after = self._get_total_min_distances()
        reward += (dist_before - dist_after) * 0.01 # Scaled distance-based reward

        # --- Check for Switch Activations ---
        for switch in self.switches:
            if not switch['active']:
                if all(drone['pos'].distance_to(switch['pos']) < self.SWITCH_ACTIVATION_DISTANCE for drone in self.drones):
                    switch['active'] = True
                    self.score += 1
                    reward += 10
                    self.activated_switch_order.append(switch['pos'])
                    self._create_particles(switch['pos'], self.COLOR_SWITCH_ACTIVE, 30)
                    # sfx: switch_activate.wav

        # --- Check for Termination ---
        terminated = False
        if self.score >= self.NUM_SWITCHES:
            # sfx: game_win.wav
            reward += 100
            terminated = True
        elif self.timer <= 0:
            # sfx: game_lose.wav
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    # --- Helper & Rendering Methods ---

    def _get_random_pos(self):
        pad = self.PLAY_AREA_PADDING
        return pygame.Vector2(
            random.uniform(pad, self.WIDTH - pad),
            random.uniform(pad, self.HEIGHT - pad)
        )

    def _clamp_drone_position(self, drone):
        pad = self.PLAY_AREA_PADDING
        drone['pos'].x = max(pad, min(self.WIDTH - pad, drone['pos'].x))
        drone['pos'].y = max(pad, min(self.HEIGHT - pad, drone['pos'].y))

    def _update_cooldowns(self):
        for drone in self.drones:
            drone['cooldown'] = max(0, drone['cooldown'] - 1)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(self.Particle(pos, color))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _get_total_min_distances(self):
        inactive_switches = [s['pos'] for s in self.switches if not s['active']]
        if not inactive_switches:
            return 0
        
        total_dist = 0
        for drone in self.drones:
            min_dist = min(drone['pos'].distance_to(sp) for sp in inactive_switches)
            total_dist += min_dist
        return total_dist

    def _render_game(self):
        # Draw Pathways
        if len(self.activated_switch_order) > 1:
            for i in range(len(self.activated_switch_order) - 1):
                p1 = self.activated_switch_order[i]
                p2 = self.activated_switch_order[i+1]
                self._draw_glowing_line(self.screen, self.COLOR_PATHWAY, p1, p2, 2, 5)

        # Draw Switches
        for switch in self.switches:
            pos = (int(switch['pos'].x), int(switch['pos'].y))
            if switch['active']:
                self._draw_glowing_circle(self.screen, self.COLOR_SWITCH_ACTIVE, pos, self.SWITCH_RADIUS, 10)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SWITCH_RADIUS, self.COLOR_SWITCH_INACTIVE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SWITCH_RADIUS, self.COLOR_SWITCH_INACTIVE)

        # Draw Particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw Drones
        for i, drone in enumerate(self.drones):
            self._draw_drone(self.screen, drone, i == self.active_drone_index)

    def _draw_drone(self, surface, drone, is_active):
        pos = drone['pos']
        color = drone['color']
        
        # Glow effect
        self._draw_glowing_circle(surface, color, (int(pos.x), int(pos.y)), self.DRONE_RADIUS + 2, 15)

        # Drone body (triangle)
        points = [
            (pos.x, pos.y - self.DRONE_RADIUS),
            (pos.x - self.DRONE_RADIUS * 0.866, pos.y + self.DRONE_RADIUS * 0.5),
            (pos.x + self.DRONE_RADIUS * 0.866, pos.y + self.DRONE_RADIUS * 0.5)
        ]
        pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], color)

        # Selection indicator
        if is_active:
            pulse = abs(math.sin(self.steps * 0.2))
            radius = int(self.DRONE_RADIUS * 1.8 + pulse * 4)
            alpha = int(100 + pulse * 100)
            self._draw_glowing_circle(surface, (255, 255, 255), (int(pos.x), int(pos.y)), radius, 5, alpha_mult=0.5)

        # Cooldown indicator
        if drone['cooldown'] > 0:
            ratio = drone['cooldown'] / self.DRONE_COOLDOWN_FRAMES
            end_angle = -math.pi / 2 + (2 * math.pi * ratio)
            rect = pygame.Rect(pos.x - self.DRONE_RADIUS, pos.y - self.DRONE_RADIUS, self.DRONE_RADIUS * 2, self.DRONE_RADIUS * 2)
            pygame.draw.arc(surface, (255,255,255), rect, -math.pi / 2, end_angle, 2)


    def _draw_glowing_circle(self, surface, color, center, radius, blur, alpha_mult=1.0):
        for i in range(blur, 0, -1):
            alpha = int(150 * (1 - i / blur)**2 * alpha_mult)
            temp_surf = pygame.Surface((radius*2 + i*2, radius*2 + i*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (radius + i, radius + i), radius + i)
            surface.blit(temp_surf, (center[0] - radius - i, center[1] - radius - i), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)


    def _draw_glowing_line(self, surface, color, start, end, width, blur):
        for i in range(blur, 0, -2):
            alpha = int(80 * (1 - i / blur)**2)
            pygame.draw.line(surface, (*color, alpha), start, end, width + i*2)
        pygame.draw.line(surface, color, start, end, width)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ACTIVATED: {self.score}/{self.NUM_SWITCHES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        time_left = f"{self.timer:.1f}"
        timer_color = self.COLOR_UI_TEXT
        if self.timer < 20: timer_color = self.COLOR_TIMER_WARN
        if self.timer < 10: timer_color = self.COLOR_TIMER_CRIT
        
        timer_text = self.font_timer.render(time_left, True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 15, 10))


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == "__main__":
    # --- Manual Play Example ---
    # This block requires a graphical display.
    # To run, comment out the os.environ line at the top of the file.
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or your display driver
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Re-enable display for manual play
    if "dummy" in os.environ.get("SDL_VIDEODRIVER", ""):
        del os.environ["SDL_VIDEODRIVER"]

    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Drone Nexus")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move selected drone")
    print("N: No-op (wait)")
    print("R: Reset environment")
    print("Q: Quit")
    
    while running:
        action_movement = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("--- Environment Reset ---")
                
                # Map keys to movement actions
                if event.key == pygame.K_UP: action_movement = 1
                elif event.key == pygame.K_DOWN: action_movement = 2
                elif event.key == pygame.K_LEFT: action_movement = 3
                elif event.key == pygame.K_RIGHT: action_movement = 4
                elif event.key == pygame.K_n: action_movement = 0
        
        # Construct the MultiDiscrete action
        # Actions 1 (space) and 2 (shift) are not used in this game
        action = [action_movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            print("--- Environment Reset ---")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()