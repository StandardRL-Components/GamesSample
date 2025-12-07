import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:15:13.354891
# Source Brief: brief_01581.md
# Brief Index: 1581
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "A futuristic bowling game where you launch energetic nuclei to cause chain reactions and knock down pins."
    user_guide = "Controls: ←→ to move, ↑↓ to aim. Press space to launch the nucleus and shift to cycle nucleus type."
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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Game Constants ---
        self.LANE_Y_START = 50
        self.LANE_Y_END = 350
        self.LANE_WIDTH = 200
        self.LANE_X_CENTER = self.screen_width // 2
        self.LANE_LEFT = self.LANE_X_CENTER - self.LANE_WIDTH // 2
        self.LANE_RIGHT = self.LANE_X_CENTER + self.LANE_WIDTH // 2
        
        self.MAX_STEPS = 2000
        self.AIM_LINE_LENGTH = 50
        self.FRICTION = 0.985
        self.SETTLE_DELAY_FRAMES = 45 # Frames to wait after ball stops

        # --- Colors ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_GRID = (30, 25, 40)
        self.COLOR_LANE = (20, 15, 30)
        self.COLOR_GUTTER = (5, 2, 10)
        self.COLOR_TEXT = (220, 220, 255)
        self.PIN_COLORS = {
            "up": (255, 80, 80), "down": (80, 80, 255), "neutral": (80, 255, 80)
        }
        self.PIN_SPINS = list(self.PIN_COLORS.keys())

        # --- Nucleus Definitions ---
        self.NUCLEUS_TYPES = [
            {"name": "Standard", "color": (255, 255, 0), "mass": 1.0, "radius": 12, "power": 12, "chain_radius": 30, "unlock_score": 0},
            {"name": "Heavy", "color": (255, 100, 255), "mass": 1.5, "radius": 15, "power": 10, "chain_radius": 35, "unlock_score": 100},
            {"name": "Chain", "color": (0, 255, 255), "mass": 0.8, "radius": 10, "power": 13, "chain_radius": 60, "unlock_score": 500},
        ]

        # --- Persistent State (survives reset) ---
        self.high_score = 0
        self.unlocked_nuclei = [n["unlock_score"] == 0 for n in self.NUCLEUS_TYPES]

        # --- Initialize state variables ---
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Episodic State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "aiming" # aiming, launched, settled
        self.settle_timer = 0
        
        # --- Game Objects ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.aim_angle = -math.pi / 2 # Pointing straight up
        self.pins = []
        self.particles = []
        
        # --- Action State ---
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Player State ---
        self.current_nucleus_index = 0

        self._setup_round()
        
        return self._get_observation(), self._get_info()

    def _setup_round(self):
        """Resets the player and pins for a new turn."""
        self.game_phase = "aiming"
        self.player_pos = pygame.Vector2(self.LANE_X_CENTER, self.LANE_Y_END - 20)
        self.player_vel = pygame.Vector2(0, 0)
        self.pins.clear()
        
        # Create 10 pins in a triangular formation
        pin_layout = [(0, 0), (-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)]
        pin_y_start = self.LANE_Y_START + 60
        pin_x_spacing = 25
        pin_y_spacing = 20
        
        for dx, dy in pin_layout:
            self.pins.append({
                "pos": pygame.Vector2(self.LANE_X_CENTER + dx * pin_x_spacing, pin_y_start + dy * pin_y_spacing),
                "vel": pygame.Vector2(0, 0),
                "spin": self.np_random.choice(self.PIN_SPINS),
                "radius": 8,
                "active": True,
                "hit_this_turn": False
            })

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        # --- Detect button presses (rising edge) ---
        launch_pressed = space_held and not self.prev_space_held
        cycle_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if cycle_pressed:
            self._cycle_nucleus()
            # sfx: UI_Switch
        
        # --- Game Phase Logic ---
        if self.game_phase == "aiming":
            self._handle_aiming(movement, launch_pressed)

        elif self.game_phase == "launched":
            reward = self._handle_launched()

        elif self.game_phase == "settled":
            self.settle_timer -= 1
            if self.settle_timer <= 0:
                self._setup_round()
        
        # --- Update game state ---
        self._update_particles()
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_aiming(self, movement, launch_pressed):
        # Move aiming position
        if movement == 3: # Left
            self.player_pos.x -= 3
        elif movement == 4: # Right
            self.player_pos.x += 3
        self.player_pos.x = np.clip(self.player_pos.x, self.LANE_LEFT + 20, self.LANE_RIGHT - 20)

        # Adjust angle
        if movement == 1: # Up
            self.aim_angle -= 0.05
        elif movement == 2: # Down
            self.aim_angle += 0.05
        self.aim_angle = np.clip(self.aim_angle, -math.pi * 0.9, -math.pi * 0.1)

        if launch_pressed:
            # sfx: Launch_Nucleus
            nucleus = self.NUCLEUS_TYPES[self.current_nucleus_index]
            self.player_vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * nucleus["power"]
            self.game_phase = "launched"

    def _handle_launched(self):
        # --- Update physics ---
        self.player_pos += self.player_vel
        self.player_vel *= self.FRICTION

        for pin in self.pins:
            if not pin["active"]: continue
            pin["pos"] += pin["vel"]
            pin["vel"] *= self.FRICTION

        # --- Handle Collisions & Rewards ---
        knocked_this_frame, is_gutter = self._handle_collisions()
        
        # --- Scoring ---
        newly_knocked_count = 0
        for pin_idx in knocked_this_frame:
            if not self.pins[pin_idx]["hit_this_turn"]:
                self.pins[pin_idx]["hit_this_turn"] = True
                newly_knocked_count += 1
        
        if newly_knocked_count > 0:
            self.score += newly_knocked_count
        
        # --- Check for end of turn ---
        ball_stopped = self.player_vel.length() < 0.1
        pins_stopped = all(p["vel"].length() < 0.1 for p in self.pins)

        if is_gutter or (ball_stopped and pins_stopped):
            self.game_phase = "settled"
            self.settle_timer = self.SETTLE_DELAY_FRAMES
            
            # --- Finalize turn rewards ---
            total_knocked_this_turn = [p for p in self.pins if p["hit_this_turn"]]
            
            is_strike = len(total_knocked_this_turn) == len(self.pins)
            is_combo = len(total_knocked_this_turn) >= 3
            
            new_highscore_achieved = self.score > self.high_score
            if new_highscore_achieved:
                self.high_score = self.score

            self._check_unlocks()
            
            return self._calculate_reward(len(total_knocked_this_turn), is_strike, is_gutter, is_combo, new_highscore_achieved)
        
        return 0.0

    def _handle_collisions(self):
        knocked_pins = set()
        is_gutter = False
        nucleus = self.NUCLEUS_TYPES[self.current_nucleus_index]
        
        # Player-wall collision (Gutter)
        if not (self.LANE_LEFT < self.player_pos.x < self.LANE_RIGHT and self.LANE_Y_START < self.player_pos.y < self.LANE_Y_END):
            is_gutter = True
            # sfx: Gutter_Ball
            self._create_particles(self.player_pos, (100, 100, 100), 20, 1)
            return set(), is_gutter

        # Player-pin collision
        for i, pin in enumerate(self.pins):
            if not pin["active"]: continue
            dist = self.player_pos.distance_to(pin["pos"])
            if dist < nucleus["radius"] + pin["radius"]:
                knocked_pins.add(i)
                # sfx: Pin_Impact
                self._create_particles(pin["pos"], self.PIN_COLORS[pin["spin"]], 30, 2)
                
                # Simple momentum transfer
                pin["vel"] = self.player_vel * (nucleus["mass"] / (nucleus["mass"] + 0.2)) * 1.1
                self.player_vel *= 0.8
                
                # Chain reaction
                self._trigger_chain_reaction(pin, knocked_pins)

        # Pin-pin collision
        for i in range(len(self.pins)):
            for j in range(i + 1, len(self.pins)):
                p1, p2 = self.pins[i], self.pins[j]
                if not p1["active"] or not p2["active"]: continue
                dist = p1["pos"].distance_to(p2["pos"])
                if dist < p1["radius"] + p2["radius"]:
                    # sfx: Pin_Clatter
                    knocked_pins.add(i)
                    knocked_pins.add(j)
                    
                    # A simple collision response
                    p1["vel"], p2["vel"] = p2["vel"] * 0.8, p1["vel"] * 0.8

        for i in knocked_pins:
            self.pins[i]["active"] = False

        return knocked_pins, is_gutter

    def _trigger_chain_reaction(self, source_pin, knocked_pins):
        nucleus = self.NUCLEUS_TYPES[self.current_nucleus_index]
        # sfx: Chain_Reaction_Start
        for i, other_pin in enumerate(self.pins):
            if not other_pin["active"]: continue
            if other_pin is source_pin: continue
            
            dist = source_pin["pos"].distance_to(other_pin["pos"])
            if dist < nucleus["chain_radius"] and other_pin["spin"] == source_pin["spin"]:
                other_pin["active"] = False
                knocked_pins.add(i)
                self._create_particles(other_pin["pos"], self.PIN_COLORS[other_pin["spin"]], 20, 1.5)

    def _calculate_reward(self, num_knocked, is_strike, is_gutter, is_combo, new_highscore):
        reward = 0.0
        reward += num_knocked * 0.1
        if is_strike: reward += 2.0 # Strike bonus
        if is_combo: reward += 1.0 # Combo bonus
        if is_gutter: reward -= 0.5 # Gutter penalty
        if new_highscore: reward += 10.0 # High score bonus
        return reward

    def _cycle_nucleus(self):
        self.current_nucleus_index = (self.current_nucleus_index + 1)
        # Skip locked nuclei
        while not self.unlocked_nuclei[self.current_nucleus_index % len(self.NUCLEUS_TYPES)]:
            self.current_nucleus_index += 1
        self.current_nucleus_index %= len(self.NUCLEUS_TYPES)

    def _check_unlocks(self):
        for i, nucleus in enumerate(self.NUCLEUS_TYPES):
            if not self.unlocked_nuclei[i] and self.score >= nucleus["unlock_score"]:
                self.unlocked_nuclei[i] = True
                # sfx: Unlock_Item

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "high_score": self.high_score}

    def _render_game(self):
        self._draw_background()
        self._draw_particles()
        self._draw_pins()
        self._draw_player()
        if self.game_phase == "aiming":
            self._draw_aim_indicator()

    def _draw_background(self):
        # Gutters
        pygame.draw.rect(self.screen, self.COLOR_GUTTER, (0, 0, self.LANE_LEFT, self.screen_height))
        pygame.draw.rect(self.screen, self.COLOR_GUTTER, (self.LANE_RIGHT, 0, self.screen_width - self.LANE_RIGHT, self.screen_height))
        # Lane
        pygame.draw.rect(self.screen, self.COLOR_LANE, (self.LANE_LEFT, 0, self.LANE_WIDTH, self.screen_height))
        # Grid lines
        for i in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.screen_height))
        for i in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.screen_width, i))

    def _draw_pins(self):
        for pin in self.pins:
            if pin["active"]:
                self._draw_glowing_circle(self.screen, self.PIN_COLORS[pin["spin"]], pin["pos"], pin["radius"])
            else: # Draw knocked-over pins as dim
                dim_color = tuple(c // 4 for c in self.PIN_COLORS[pin["spin"]])
                pygame.gfxdraw.filled_circle(self.screen, int(pin["pos"].x), int(pin["pos"].y), pin["radius"], dim_color)
                
    def _draw_player(self):
        nucleus = self.NUCLEUS_TYPES[self.current_nucleus_index]
        self._draw_glowing_circle(self.screen, nucleus["color"], self.player_pos, nucleus["radius"])

    def _draw_aim_indicator(self):
        end_pos = self.player_pos + pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * self.AIM_LINE_LENGTH
        pygame.draw.line(self.screen, (255, 255, 255, 150), self.player_pos, end_pos, 2)

    def _draw_glowing_circle(self, surface, color, pos, radius):
        glow_radius = int(radius * 2.5)
        glow_alpha = 40
        
        # Create a temporary surface for the glow
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        
        surface.blit(glow_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), radius, color)
        pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), radius, color)

    def _create_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(20, 40),
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 40))
            size = max(1, int(3 * (p["lifespan"] / 40)))
            pygame.draw.circle(self.screen, (*p["color"], alpha), p["pos"], size)

    def _render_ui(self):
        # Score and High Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        highscore_text = self.font_small.render(f"BEST: {self.high_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(highscore_text, (10, 30))

        # Current Nucleus
        nucleus = self.NUCLEUS_TYPES[self.current_nucleus_index]
        nucleus_name_text = self.font_small.render(f"NUCLEUS:", True, self.COLOR_TEXT)
        nucleus_type_text = self.font_large.render(f"{nucleus['name']}", True, nucleus['color'])
        self.screen.blit(nucleus_name_text, (self.screen_width - 150, self.screen_height - 50))
        self.screen.blit(nucleus_type_text, (self.screen_width - 150, self.screen_height - 35))

        # Phase display for debugging/clarity
        # phase_text = self.font_small.render(f"PHASE: {self.game_phase}", True, self.COLOR_TEXT)
        # self.screen.blit(phase_text, (self.screen_width/2 - 50, 10))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we don't want the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Boson Bowling")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated:
            print("Episode finished!")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()