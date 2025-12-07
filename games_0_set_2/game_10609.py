import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:41:31.674484
# Source Brief: brief_00609.md
# Brief Index: 609
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a submersible through layered ocean depths, strategically opening and 
    closing portals to manage pressure and collect rare resources while protecting your base.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a submersible through ocean depths. Open and close portals to manage "
        "pressure, collect resources, and protect your base from being crushed."
    )
    user_guide = (
        "Use the ↑ and ↓ arrow keys to select a portal. Press space to open or close the selected portal."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.NUM_LAYERS = 8
        self.MAX_STEPS = 2000
        self.BASE_PRESSURE_THRESHOLD = 50.0
        self.BASE_MAX_HEALTH = 100.0

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 160, 0)
        self.COLOR_RESOURCE = (0, 255, 150)
        self.COLOR_RESOURCE_GLOW = (0, 200, 120)
        self.COLOR_PORTAL_CLOSED = (100, 110, 130)
        self.COLOR_PORTAL_OPEN = (255, 255, 100)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (40, 200, 80)
        self.COLOR_HEALTH_DAMAGE = (220, 50, 50)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.base_health = 0.0
        self.pressures = []
        self.ambient_pressures = []
        self.portals = []
        self.resources = []
        self.particles = []
        self.player_layer_idx = 0
        self.player_visual_y = 0
        self.max_depth_reached = 0
        self.selected_portal_idx = 0
        self.prev_space_held = False
        self.collected_this_step = False
        
        # This is a bit unusual, but the original code called reset() and validate() here.
        # We'll keep it to maintain behavior, though typically reset() is called by the user.
        # self.reset()
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.base_health = self.BASE_MAX_HEALTH
        
        # Initialize pressures
        self.ambient_pressures = [10.0 + i * 15.0 for i in range(self.NUM_LAYERS)]
        self.pressures = list(self.ambient_pressures)
        
        # Initialize portals
        self.portals = [{'is_open': False, 'anim_state': 0.0} for _ in range(self.NUM_LAYERS - 1)]
        self.selected_portal_idx = 0
        
        # Initialize resources
        self.resources = []
        resource_layers = random.sample(range(1, self.NUM_LAYERS), 4)
        resource_layers.sort()
        for i, layer_idx in enumerate(resource_layers):
            is_deepest = i == len(resource_layers) - 1
            self.resources.append({
                'layer_idx': layer_idx, 
                'is_collected': False, 
                'is_deepest': is_deepest,
                'pulse': random.random() * math.pi * 2
            })
            
        # Initialize player
        self.player_layer_idx = 0
        self.player_visual_y = self._get_y_for_layer(0)
        self.max_depth_reached = 0
        
        self.particles = []
        self.prev_space_held = True # Prevent toggle on first step
        self.collected_this_step = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Actions
        self._handle_actions(movement, space_held)
        
        # 2. Update Game Logic
        self._update_portal_animations()
        self._update_pressure()
        self._update_base_health()
        self._update_player_position()
        self._update_particles()
        
        # 3. Check for events
        self.collected_this_step = self._check_resource_collection()
        self._update_difficulty()
        
        # 4. Calculate Reward
        reward = self._calculate_reward()
        self.score += reward
        
        # 5. Check Termination
        terminated = self._check_termination()
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Action and Logic Updates ---

    def _handle_actions(self, movement, space_held):
        # Movement action selects a portal
        if movement == 1: # Up
            self.selected_portal_idx = max(0, self.selected_portal_idx - 1)
        elif movement == 2: # Down
            self.selected_portal_idx = min(self.NUM_LAYERS - 2, self.selected_portal_idx + 1)
        
        # Space action toggles the selected portal
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # sfx: portal_toggle.wav
            self.portals[self.selected_portal_idx]['is_open'] = not self.portals[self.selected_portal_idx]['is_open']
        self.prev_space_held = space_held

    def _update_portal_animations(self):
        for portal in self.portals:
            target = 1.0 if portal['is_open'] else 0.0
            portal['anim_state'] += (target - portal['anim_state']) * 0.2

    def _update_pressure(self):
        pressure_flow_rate = 0.1
        ambient_revert_rate = 0.005
        
        # Calculate flow through open portals
        deltas = [0.0] * self.NUM_LAYERS
        for i, portal in enumerate(self.portals):
            if portal['is_open']:
                p1, p2 = self.pressures[i], self.pressures[i+1]
                diff = p1 - p2
                flow = diff * pressure_flow_rate
                deltas[i] -= flow
                deltas[i+1] += flow
                
                # Spawn particles for visual feedback
                if abs(flow) > 0.1:
                    self._spawn_pressure_particles(i, flow)

        # Apply deltas and revert to ambient pressure
        for i in range(self.NUM_LAYERS):
            self.pressures[i] += deltas[i]
            revert_flow = (self.ambient_pressures[i] - self.pressures[i]) * ambient_revert_rate
            self.pressures[i] += revert_flow
            self.pressures[i] = max(0, self.pressures[i])

    def _update_base_health(self):
        pressure_on_base = self.pressures[0]
        if pressure_on_base > self.BASE_PRESSURE_THRESHOLD:
            # sfx: alarm_loop.wav, hull_creak.wav
            damage = (pressure_on_base - self.BASE_PRESSURE_THRESHOLD) * 0.2
            self.base_health = max(0, self.base_health - damage)

    def _update_player_position(self):
        # Find all layers reachable from the player's current layer
        q = [self.player_layer_idx]
        visited = {self.player_layer_idx}
        head = 0
        while head < len(q):
            curr = q[head]
            head += 1
            # Check portal above
            if curr > 0 and self.portals[curr - 1]['is_open'] and (curr - 1) not in visited:
                visited.add(curr - 1)
                q.append(curr - 1)
            # Check portal below
            if curr < self.NUM_LAYERS - 1 and self.portals[curr]['is_open'] and (curr + 1) not in visited:
                visited.add(curr + 1)
                q.append(curr + 1)
        
        # Find the layer with the minimum pressure among reachable layers
        min_pressure = float('inf')
        best_layer = self.player_layer_idx
        for layer_idx in visited:
            if self.pressures[layer_idx] < min_pressure:
                min_pressure = self.pressures[layer_idx]
                best_layer = layer_idx
        
        self.player_layer_idx = best_layer
        
        # Update visual position smoothly
        target_y = self._get_y_for_layer(self.player_layer_idx)
        self.player_visual_y += (target_y - self.player_visual_y) * 0.1

    def _check_resource_collection(self):
        for res in self.resources:
            if not res['is_collected'] and res['layer_idx'] == self.player_layer_idx:
                res['is_collected'] = True
                # sfx: resource_collect.wav
                return True
        return False

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            for i in range(self.NUM_LAYERS):
                self.ambient_pressures[i] += 0.5

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.05)

    def _spawn_pressure_particles(self, portal_idx, flow):
        if random.random() < 0.7: # Don't spawn every frame
            return
        portal_y = self._get_y_for_layer(portal_idx + 0.5)
        portal_x = self.SCREEN_WIDTH / 2
        for _ in range(2):
            vel_y = math.copysign(random.uniform(0.5, 1.5), -flow)
            self.particles.append({
                'pos': [portal_x + random.uniform(-15, 15), portal_y],
                'vel': [random.uniform(-0.2, 0.2), vel_y],
                'life': random.randint(20, 40),
                'size': random.uniform(2, 4),
                'color': (random.randint(150, 200), random.randint(200, 255), 255)
            })

    # --- Reward and Termination ---

    def _calculate_reward(self):
        reward = 0.0
        
        # Reward for reaching new depths
        if self.player_layer_idx > self.max_depth_reached:
            reward += (self.player_layer_idx - self.max_depth_reached) * 0.1
            self.max_depth_reached = self.player_layer_idx
        
        # Penalty for base taking damage
        if self.base_health < self.BASE_MAX_HEALTH:
            reward -= 0.1
        
        # Reward for collecting resources
        if self.collected_this_step:
            reward += 5.0
            
        return reward

    def _check_termination(self):
        terminated = False
        if self.base_health <= 0:
            terminated = True
            self.score -= 100.0
            self.game_over = True
            self.game_over_message = "BASE DESTROYED"
        
        deepest_resource = next(res for res in self.resources if res['is_deepest'])
        if deepest_resource['is_collected']:
            terminated = True
            self.score += 100.0
            self.game_over = True
            self.game_over_message = "VICTORY"

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME LIMIT REACHED"
            
        return terminated

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_layers()
        self._render_portals()
        self._render_particles()
        self._render_resources()
        self._render_player()
        self._render_base()

    def _render_layers(self):
        for i in range(self.NUM_LAYERS):
            y_start = self._get_y_for_layer(i)
            y_end = self._get_y_for_layer(i + 1)
            
            # Darker with depth
            lerp_factor = i / (self.NUM_LAYERS - 1)
            color = self._lerp_color((30, 40, 70), (5, 10, 20), lerp_factor)
            pygame.draw.rect(self.screen, color, (0, y_start, self.SCREEN_WIDTH, y_end - y_start))

    def _render_portals(self):
        portal_x = self.SCREEN_WIDTH / 2
        for i, portal in enumerate(self.portals):
            portal_y = self._get_y_for_layer(i + 1)
            
            # Draw selector
            if i == self.selected_portal_idx:
                pygame.gfxdraw.filled_circle(self.screen, int(portal_x), int(portal_y), 14, (*self.COLOR_SELECTOR, 50))
                pygame.gfxdraw.aacircle(self.screen, int(portal_x), int(portal_y), 14, self.COLOR_SELECTOR)

            # Draw portal
            color = self._lerp_color(self.COLOR_PORTAL_CLOSED, self.COLOR_PORTAL_OPEN, portal['anim_state'])
            radius = 3 + 7 * portal['anim_state']
            if radius > 0:
                pygame.draw.circle(self.screen, color, (portal_x, portal_y), radius)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            if alpha > 0:
                color = (*p['color'], alpha)
                surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_resources(self):
        for res in self.resources:
            if not res['is_collected']:
                y = self._get_y_for_layer(res['layer_idx'] + 0.5)
                x = self.SCREEN_WIDTH * 0.75
                res['pulse'] += 0.1
                size_pulse = 1 + math.sin(res['pulse']) * 0.2
                self._draw_glowing_star(x, y, self.COLOR_RESOURCE, self.COLOR_RESOURCE_GLOW, size_pulse)

    def _render_player(self):
        x = self.SCREEN_WIDTH / 4
        self._draw_glowing_circle(x, self.player_visual_y, 12, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        # "Cockpit"
        pygame.draw.circle(self.screen, (200, 255, 255), (x + 4, self.player_visual_y - 2), 3)

    def _render_base(self):
        base_y = self._get_y_for_layer(0)
        base_x = self.SCREEN_WIDTH / 4
        pygame.draw.rect(self.screen, (80, 90, 110), (base_x - 25, base_y - 15, 50, 15))
        pygame.gfxdraw.filled_circle(self.screen, int(base_x), int(base_y - 15), 20, (120, 130, 150))
        pygame.gfxdraw.aacircle(self.screen, int(base_x), int(base_y - 15), 20, (150, 160, 180))

    def _render_ui(self):
        # Pressure gauges
        for i in range(self.NUM_LAYERS):
            y = self._get_y_for_layer(i + 0.5)
            pressure_val = self.pressures[i]
            
            # Color based on pressure
            max_p = self.ambient_pressures[-1] + 20
            norm_p = min(1.0, pressure_val / max_p)
            color = self._lerp_color((0, 150, 255), (255, 50, 50), norm_p**2)
            
            self._draw_text(f"{pressure_val:.1f}", (self.SCREEN_WIDTH - 50, y), color=color, center_y=True)

        # Base Health
        health_pct = self.base_health / self.BASE_MAX_HEALTH
        health_bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_DAMAGE, (20, 20, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (20, 20, health_bar_width * health_pct, 20))
        self._draw_text("BASE HEALTH", (20 + health_bar_width/2, 30), center_x=True, center_y=True)

        # Resources collected
        collected_count = sum(1 for r in self.resources if r['is_collected'])
        total_count = len(self.resources)
        self._draw_text(f"RESOURCES: {collected_count}/{total_count}", (self.SCREEN_WIDTH - 20, 20), anchor="topright")
        
        # Depth Meter
        depth_meter_x = 30
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (depth_meter_x, 50), (depth_meter_x, self.SCREEN_HEIGHT - 20), 2)
        indicator_y = 50 + (self.player_visual_y - self._get_y_for_layer(0)) / (self._get_y_for_layer(self.NUM_LAYERS) - self._get_y_for_layer(0)) * (self.SCREEN_HEIGHT - 70)
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(depth_meter_x, indicator_y), (depth_meter_x-10, indicator_y-5), (depth_meter_x-10, indicator_y+5)])

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        is_win = "VICTORY" in self.game_over_message
        color = self.COLOR_RESOURCE if is_win else self.COLOR_HEALTH_DAMAGE
        self._draw_text(self.game_over_message, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), font=self.font_l, color=color, center_x=True, center_y=True)
        self._draw_text(f"Final Score: {self.score:.1f}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30), font=self.font_m, center_x=True, center_y=True)

    # --- Helper Methods ---

    def _get_y_for_layer(self, layer_idx):
        return 50 + (self.SCREEN_HEIGHT - 70) * (layer_idx / self.NUM_LAYERS)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health}

    def _draw_text(self, text, pos, font=None, color=None, center_x=False, center_y=False, anchor="topleft"):
        font = font or self.font_s
        color = color or self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        x, y = pos
        if center_x: x -= text_rect.width / 2
        if center_y: y -= text_rect.height / 2
        if anchor == "topright": x -= text_rect.width
        
        self.screen.blit(text_surface, (x, y))

    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        return (c1[0] + (c2[0] - c1[0]) * t, c1[1] + (c2[1] - c1[1]) * t, c1[2] + (c2[2] - c1[2]) * t)

    def _draw_glowing_circle(self, x, y, radius, color, glow_color):
        for i in range(4):
            alpha = 80 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(radius + i * 2), (*glow_color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), radius, color)
    
    def _draw_glowing_star(self, x, y, color, glow_color, scale):
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            r = (8 if i % 2 == 0 else 4) * scale
            points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
        
        # Glow
        for i in range(3):
            alpha = 60 - i * 15
            glow_points = [(p[0] + (x-p[0]) * i * -0.2, p[1] + (y-p[1]) * i * -0.2) for p in points]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*glow_color, alpha))
        
        # Star
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the headless environment variable
    # and use a display-enabled pygame backend.
    # For example:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv()
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # Pygame setup for manual play
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Submersible Pressure Manager")
        clock = pygame.time.Clock()
        display_enabled = True
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        display_enabled = False

    
    movement = 0
    space_held = False
    
    print("Controls: UP/DOWN arrows to select portal, SPACE to toggle.")
    print("Goal: Collect the deepest resource without your base being crushed.")

    while not terminated:
        if display_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    if event.key == pygame.K_SPACE:
                        space_held = True
                if event.type == pygame.KEYUP:
                    if event.key in [pygame.K_UP, pygame.K_DOWN]:
                        movement = 0
                    if event.key == pygame.K_SPACE:
                        space_held = False
        else: # Simple auto-play for headless mode
            action = env.action_space.sample()
            movement, space_held = action[0], action[1] == 1
            if env.steps > 1000: # Stop headless run after some steps
                terminated = True

        
        action = [movement, 1 if space_held else 0, 0]
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term or terminated
        
        # Render the observation to the display
        if display_enabled:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30) # Run at 30 FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            if display_enabled:
                # Wait a bit before closing
                pygame.time.wait(3000)

    pygame.quit()