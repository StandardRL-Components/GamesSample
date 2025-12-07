import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:30:38.427936
# Source Brief: brief_00501.md
# Brief Index: 501
# """import gymnasium as gym
class GameEnv(gym.Env):
    """
    GameEnv: Manipulate valves to control fluid flow, transform its state,
    and fill containers to specific volumes within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate a series of valves to control fluid flow and its state. "
        "Fill containers to their target levels with the correct liquid form before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select a valve. Press space to open or close the selected valve."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game Configuration ---
        self.max_steps = 1800  # 60 seconds at 30 FPS
        self.flow_rate = 5.0  # Units per step

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PIPE = (60, 70, 90)
        self.COLOR_PIPE_FLOW = (80, 120, 150)
        self.COLOR_LIQUID = (50, 150, 255)
        self.COLOR_SOLID = (220, 220, 255)
        self.COLOR_GAS = (180, 190, 200)
        self.COLOR_VALVE_CLOSED = (200, 80, 80)
        self.COLOR_VALVE_OPEN = (80, 200, 80)
        self.COLOR_SELECTION = (255, 220, 50)
        self.COLOR_TARGET_LINE = (255, 100, 0, 150)
        self.COLOR_TARGET_MET = (100, 255, 100)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TIMER_URGENT = (255, 60, 60)

        # --- Game Element Layout ---
        self._define_layout()
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.valves = []
        self.containers = []
        self.fluid_state = "liquid"
        self.selected_valve_index = 0
        self.last_space_state = False
        self.last_movement_action = 0
        self.particles = []
        self.achieved_targets = []
        
        # self.reset() is called by the environment wrapper
        
    def _define_layout(self):
        self.layout = {
            "source": (320, 40),
            "v1": (320, 80),
            "j1": (320, 120),
            "v2": (160, 160),
            "c1": (160, 280),
            "v3": (480, 160),
            "j2": (480, 200),
            "v4": (400, 240),
            "c2": (400, 320),
            "v5": (560, 240),
            "c3": (560, 320),
        }
        
        self.pipes = [
            (self.layout["source"], self.layout["v1"]),
            (self.layout["v1"], self.layout["j1"]),
            (self.layout["j1"], self.layout["v2"]),
            (self.layout["v2"], self.layout["c1"]),
            (self.layout["j1"], self.layout["v3"]),
            (self.layout["v3"], self.layout["j2"]),
            (self.layout["j2"], self.layout["v4"]),
            (self.layout["v4"], self.layout["c2"]),
            (self.layout["j2"], self.layout["v5"]),
            (self.layout["v5"], self.layout["c3"]),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fluid_state = "liquid"
        self.selected_valve_index = 0
        self.last_space_state = False
        self.last_movement_action = 0
        self.particles.clear()
        
        self.valves = [
            {"pos": self.layout["v1"], "open": True, "size": (60, 20)},
            {"pos": self.layout["v2"], "open": False, "size": (60, 20)},
            {"pos": self.layout["v3"], "open": False, "size": (60, 20)},
            {"pos": self.layout["v4"], "open": False, "size": (60, 20)},
            {"pos": self.layout["v5"], "open": False, "size": (60, 20)},
        ]

        self.containers = [
            {"pos": self.layout["c1"], "size": (80, 120), "capacity": 120, "target": 100, "level": 0, "visual_level": 0, "fluid_type": "none"},
            {"pos": self.layout["c2"], "size": (80, 120), "capacity": 170, "target": 150, "level": 0, "visual_level": 0, "fluid_type": "none"},
            {"pos": self.layout["c3"], "size": (80, 120), "capacity": 220, "target": 200, "level": 0, "visual_level": 0, "fluid_type": "none"},
        ]
        self.achieved_targets = [False, False, False]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty

        self._handle_input(action)
        step_reward = self._update_game_state()
        reward += step_reward
        
        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        if self.game_over:
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_action, _ = action
        space_held = space_held_action == 1

        # Handle valve selection cycling
        if movement != self.last_movement_action and movement != 0:
            if movement in [1, 3]:  # Up or Left
                self.selected_valve_index = (self.selected_valve_index - 1) % len(self.valves)
                # sfx: ui_blip_low
            elif movement in [2, 4]:  # Down or Right
                self.selected_valve_index = (self.selected_valve_index + 1) % len(self.valves)
                # sfx: ui_blip_high
        self.last_movement_action = movement

        # Handle valve toggle on rising edge of space press
        if space_held and not self.last_space_state:
            self.valves[self.selected_valve_index]["open"] = not self.valves[self.selected_valve_index]["open"]
            # sfx: valve_toggle_click
        self.last_space_state = space_held

    def _update_game_state(self):
        reward = 0

        # 1. Determine current fluid state based on valve combinations
        v2_open = self.valves[1]["open"]
        v3_open = self.valves[2]["open"]
        v4_open = self.valves[3]["open"]
        v5_open = self.valves[4]["open"]

        if v2_open and v3_open:
            self.fluid_state = "gas" # sfx: system_overheat_hiss
        elif v4_open and v5_open:
            self.fluid_state = "solid" # sfx: system_freeze_crackle
        else:
            self.fluid_state = "liquid"

        # 2. Simulate fluid flow
        flows = {"v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0}
        if self.valves[0]["open"]:
            flows["v1"] = self.flow_rate
        
        # Split at Junction 1
        if v2_open and v3_open:
            flows["v2"] = flows["v1"] / 2
            flows["v3"] = flows["v1"] / 2
        elif v2_open:
            flows["v2"] = flows["v1"]
        elif v3_open:
            flows["v3"] = flows["v1"]
        
        # Split at Junction 2
        if v4_open and v5_open:
            flows["v4"] = flows["v3"] / 2
            flows["v5"] = flows["v3"] / 2
        elif v4_open:
            flows["v4"] = flows["v3"]
        elif v5_open:
            flows["v5"] = flows["v3"]
        
        # 3. Add fluid to containers and calculate reward
        container_map = {1: flows["v2"], 3: flows["v4"], 4: flows["v5"]}
        for valve_idx, flow in container_map.items():
            if flow > 0:
                container_idx = {1: 0, 3: 1, 4: 2}[valve_idx]
                reward += self._add_fluid_to_container(container_idx, flow)
        
        # 4. Update visual levels (lerp for smooth animation)
        for c in self.containers:
            c["visual_level"] += (c["level"] - c["visual_level"]) * 0.2

        # 5. Update particles
        self._update_particles(flows)
        
        self.score += reward
        return reward

    def _add_fluid_to_container(self, idx, amount):
        container = self.containers[idx]
        
        if container["level"] >= container["capacity"]:
            return -0.05 # Penalty for trying to overfill
        
        # If container is empty or contains the same fluid, add it
        if container["fluid_type"] == "none" or container["fluid_type"] == self.fluid_state:
            container["fluid_type"] = self.fluid_state
            added_amount = min(amount, container["capacity"] - container["level"])
            container["level"] += added_amount
            
            if self.fluid_state == "liquid":
                return added_amount * 0.02 # Reward for correct fluid
            else:
                return -added_amount * 0.05 # Penalty for wrong fluid
        
        # If fluid type is different, it's a mistake
        else:
            return -amount * 0.1 # Big penalty for mixing fluids

    def _check_termination(self):
        # Check for newly met targets
        all_targets_met = True
        newly_met_reward = 0
        for i, c in enumerate(self.containers):
            if c["level"] >= c["target"] and c["fluid_type"] == "liquid":
                if not self.achieved_targets[i]:
                    self.achieved_targets[i] = True
                    # sfx: target_met_chime
                    newly_met_reward += 25.0 # Big reward for meeting a target
            else:
                all_targets_met = False
        
        if newly_met_reward > 0:
            return False, newly_met_reward
            
        # Win condition
        if all_targets_met:
            # sfx: win_fanfare
            return True, 100.0
        
        # Lose condition
        if self.steps >= self.max_steps:
            # sfx: lose_buzzer
            return True, -50.0
            
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fluid_state": self.fluid_state,
            "container_levels": [c["level"] for c in self.containers],
        }

    def _render_game(self):
        self._render_pipes()
        self._render_particles()
        self._render_containers()
        self._render_valves()
        
        # Source Tank
        pygame.draw.rect(self.screen, self.COLOR_PIPE, (280, 20, 80, 30), border_radius=5)
        source_text = self.font_small.render("SOURCE", True, self.COLOR_TEXT)
        self.screen.blit(source_text, (295, 25))

    def _render_pipes(self):
        for start, end in self.pipes:
            pygame.draw.line(self.screen, self.COLOR_PIPE, start, end, 12)
            pygame.draw.line(self.screen, self.COLOR_PIPE_FLOW, start, end, 8)

    def _render_valves(self):
        for i, v in enumerate(self.valves):
            rect = pygame.Rect(0, 0, v["size"][0], v["size"][1])
            rect.center = v["pos"]
            
            # Selection Highlight
            if i == self.selected_valve_index:
                self._draw_glowing_rect(rect, self.COLOR_SELECTION, 15)

            pygame.draw.rect(self.screen, self.COLOR_PIPE, rect, border_radius=4)
            
            inner_color = self.COLOR_VALVE_OPEN if v["open"] else self.COLOR_VALVE_CLOSED
            inner_rect = rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, inner_color, inner_rect, border_radius=3)
            
            valve_text = self.font_small.render(f"V{i+1}", True, self.COLOR_TEXT)
            self.screen.blit(valve_text, (rect.centerx - 10, rect.centery - 35))

    def _render_containers(self):
        for i, c in enumerate(self.containers):
            base_rect = pygame.Rect(0, 0, c["size"][0], c["size"][1])
            base_rect.midbottom = c["pos"]
            
            # Border changes color if target is met
            border_color = self.COLOR_TARGET_MET if self.achieved_targets[i] else self.COLOR_PIPE
            if self.achieved_targets[i]:
                self._draw_glowing_rect(base_rect, border_color, 20)
            pygame.draw.rect(self.screen, border_color, base_rect, 4, border_radius=5)
            
            # Fluid inside
            if c["level"] > 0:
                fluid_height = (c["visual_level"] / c["capacity"]) * c["size"][1]
                fluid_rect = pygame.Rect(
                    base_rect.left,
                    base_rect.bottom - fluid_height,
                    c["size"][0],
                    fluid_height
                )
                fluid_color = {
                    "liquid": self.COLOR_LIQUID,
                    "solid": self.COLOR_SOLID,
                    "gas": self.COLOR_GAS
                }[c["fluid_type"]]
                pygame.draw.rect(self.screen, fluid_color, fluid_rect.clip(base_rect), border_bottom_left_radius=5, border_bottom_right_radius=5)

            # Target line
            target_y = base_rect.bottom - (c["target"] / c["capacity"]) * c["size"][1]
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (base_rect.left, target_y), (base_rect.right, target_y), 2)
            
            # Text display
            level_text = f"{int(c['level'])} / {c['target']}"
            text_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (base_rect.centerx - text_surf.get_width() // 2, base_rect.bottom + 5))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.max_steps - self.steps) / 30.0
        timer_color = self.COLOR_TIMER_URGENT if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"{time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 10, 10))
        
        # Current Fluid State
        state_color = {"liquid": self.COLOR_LIQUID, "solid": self.COLOR_SOLID, "gas": self.COLOR_GAS}[self.fluid_state]
        state_text = self.font_medium.render(f"State: {self.fluid_state.upper()}", True, state_color)
        self.screen.blit(state_text, (10, 40))

    def _update_particles(self, flows):
        # Create new particles
        flow_map = {0:"v1", 1:"v1", 2:"v2", 3:"v2", 4:"v3", 5:"v3", 6:"v4", 7:"v4", 8:"v5", 9:"v5"}

        for i, (start, end) in enumerate(self.pipes):
            flow_key = flow_map.get(i)
            if flow_key and flows[flow_key] > 0 and self.np_random.random() < 0.5:
                pos = list(start)
                direction = (end[0] - start[0], end[1] - start[1])
                dist = math.hypot(*direction)
                if dist == 0: continue
                vel = [(c / dist) * 3 for c in direction]
                
                p_color = {"liquid": self.COLOR_LIQUID, "solid": self.COLOR_SOLID, "gas": self.COLOR_GAS}[self.fluid_state]
                p_size = 4 if self.fluid_state == "solid" else 2
                p_lifespan = dist / 3
                
                self.particles.append({"pos": pos, "vel": vel, "lifespan": p_lifespan, "color": p_color, "size": p_size})

        # Update and remove old particles
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), p["size"], p["color"])
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), p["size"], p["color"])
            
    def _draw_glowing_rect(self, rect, color, radius):
        for i in range(radius, 0, -2):
            alpha = 255 * (1 - i / radius)
            glow_color = (*color, int(alpha))
            glow_rect = rect.inflate(i, i)
            pygame.draw.rect(self.screen, glow_color, glow_rect, 2, border_radius=int(i/2)+5)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- This is for human play ---
    # It's not used by the evaluation system,
    # but is helpful for testing and debugging.
    
    # Un-set the headless environment variable to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Manual Play Controls ---
    # Arrows: Select Valve
    # Space: Toggle Valve
    # Q: Quit
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    # Create a display
    display_surface = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Fluid Dynamics Puzzle")

    keys_down = set()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                keys_down.add(event.key)
            if event.type == pygame.KEYUP:
                if event.key in keys_down:
                    keys_down.remove(event.key)

        # Map keys to MultiDiscrete action
        # This mapping is for continuous key presses, suitable for auto_advance=True
        action.fill(0)
        if pygame.K_UP in keys_down: action[0] = 1
        elif pygame.K_DOWN in keys_down: action[0] = 2
        elif pygame.K_LEFT in keys_down: action[0] = 3
        elif pygame.K_RIGHT in keys_down: action[0] = 4
        
        if pygame.K_SPACE in keys_down: action[1] = 1
        else: action[1] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render for human viewing
        render_surface = pygame.transform.rotate(pygame.surfarray.make_surface(obs), -90)
        render_surface = pygame.transform.flip(render_surface, True, False)
        display_surface.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total reward: {total_reward}")
            print("Resetting environment...")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        env.clock.tick(30) # Run at 30 FPS for smooth visuals
        
    env.close()