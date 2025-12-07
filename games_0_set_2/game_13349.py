import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:55:11.565421
# Source Brief: brief_03349.md
# Brief Index: 3349
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a buoyant submarine to the surface.

    The player must manage the submarine's buoyancy to ascend while navigating
    treacherous currents and collecting energy orbs. The goal is to reach the
    surface (y=0) within 60 seconds without letting buoyancy levels become
    critical.

    **Visuals:**
    The environment features a visually polished 2D presentation with a deep-sea
    aesthetic. Elements include:
    - A dynamic background gradient from deep to light blue.
    - A player-controlled submarine that tilts and emits bubbles.
    - Animated, swirling water currents that affect the submarine.
    - Glowing energy orbs that provide a speed boost.
    - A clear UI displaying depth, time, buoyancy, and collected orbs.

    **Physics and Gameplay:**
    - Vertical movement is controlled indirectly by adjusting buoyancy.
    - Horizontal movement is controlled by direct thruster input.
    - The submarine has momentum, and water drag provides a dampening effect.
    - Currents apply a force and drain buoyancy on contact.
    - Orbs grant a temporary speed boost.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]` (Movement): 0=none, 1=increase buoyancy, 2=decrease buoyancy, 3=thrust left, 4=thrust right.
    - `actions[1]` (Space): Unused.
    - `actions[2]` (Shift): Unused.

    **Reward Structure:**
    - Positive rewards for ascending, moving towards ideal buoyancy (60%), and collecting orbs.
    - Negative rewards for descending and deviating from ideal buoyancy.
    - Large terminal rewards for winning (+100) or losing (-100).
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pilot a buoyant submarine to the surface, managing its ascent while navigating "
        "treacherous currents and collecting energy orbs."
    )
    user_guide = (
        "Use ↑ to increase buoyancy, ↓ to decrease it, and ←→ to thrust horizontally. "
        "Reach the surface before time runs out."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG_TOP = (20, 40, 80)
    COLOR_BG_BOTTOM = (5, 10, 20)
    COLOR_SURFACE = (200, 220, 255)
    COLOR_PLAYER = (255, 180, 0)
    COLOR_PLAYER_GLOW = (255, 180, 0, 40)
    COLOR_ORB = (255, 255, 0)
    COLOR_ORB_GLOW = (255, 255, 0, 60)
    COLOR_CURRENT = (100, 120, 200, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BAR_BG = (50, 50, 50)
    COLOR_UI_BAR_GOOD = (0, 200, 0)
    COLOR_UI_BAR_WARN = (255, 255, 0)
    COLOR_UI_BAR_BAD = (200, 0, 0)

    # Player Physics
    PLAYER_THRUST = 0.3
    BUOYANCY_SENSITIVITY = 0.15
    BUOYANCY_FORCE_FACTOR = -0.008  # Negative because y=0 is up
    DRAG_COEFFICIENT = 0.97
    PLAYER_TILT_FACTOR = 3.0
    BASE_MAX_SPEED = 4.0

    # Game Mechanics
    NUM_ORBS = 10
    NUM_CURRENTS = 3
    CURRENT_FORCE = -0.05  # Buoyancy drain per step in contact
    CURRENT_BASE_SPEED = 0.5
    ORB_SPEED_BOOST_DURATION = 10 * FPS
    ORB_SPEED_BOOST_AMOUNT = 2.0

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
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sub_pos = np.array([0.0, 0.0])
        self.sub_vel = np.array([0.0, 0.0])
        self.buoyancy = 0.0
        self.orbs_collected = 0
        self.currents = []
        self.orbs = []
        self.particles = []
        self.speed_boost_timer = 0
        self.last_y = 0
        self.last_buoyancy_dist = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.sub_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50.0])
        self.sub_vel = np.array([0.0, 0.0])
        self.buoyancy = 60.0
        self.orbs_collected = 0
        self.speed_boost_timer = 0
        
        self.last_y = self.sub_pos[1]
        self.last_buoyancy_dist = abs(self.buoyancy - 60.0)

        self.orbs = [
            np.array([
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(50, self.HEIGHT - 80),
            ])
            for _ in range(self.NUM_ORBS)
        ]

        self.currents = []
        for _ in range(self.NUM_CURRENTS):
            self.currents.append(self._create_current())

        self.particles = []

        return self._get_observation(), self._get_info()

    def _create_current(self):
        return {
            "pos": np.array([
                self.np_random.uniform(0, self.WIDTH),
                self.np_random.uniform(0, self.HEIGHT),
            ]),
            "vel": self.np_random.uniform(-1, 1, size=2),
            "radius": self.np_random.uniform(40, 70),
            "anim_offset": self.np_random.uniform(0, 2 * math.pi)
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0
        self.steps += 1

        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_physics()
        self._update_entities()
        
        reward += self._handle_collisions()
        reward += self._calculate_reward()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.sub_pos[1] <= 10: # Win condition
                reward += 100
            else: # Loss condition
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement):
        # Buoyancy control
        if movement == 1:  # Up
            self.buoyancy += self.BUOYANCY_SENSITIVITY
            # sfx: buoyancy pump high
        elif movement == 2:  # Down
            self.buoyancy -= self.BUOYANCY_SENSITIVITY
            # sfx: buoyancy pump low

        self.buoyancy = np.clip(self.buoyancy, 0, 100)

        # Horizontal thrust
        if movement == 3:  # Left
            self.sub_vel[0] -= self.PLAYER_THRUST
            self._spawn_particles(angle_offset=0)
        elif movement == 4:  # Right
            self.sub_vel[0] += self.PLAYER_THRUST
            self._spawn_particles(angle_offset=math.pi)

    def _update_physics(self):
        # Apply buoyancy force
        buoyancy_diff = self.buoyancy - 60.0
        vertical_force = buoyancy_diff * self.BUOYANCY_FORCE_FACTOR
        self.sub_vel[1] += vertical_force

        # Apply drag
        self.sub_vel *= self.DRAG_COEFFICIENT

        # Clamp velocity
        max_speed = self.BASE_MAX_SPEED
        if self.speed_boost_timer > 0:
            max_speed += self.ORB_SPEED_BOOST_AMOUNT
        speed = np.linalg.norm(self.sub_vel)
        if speed > max_speed:
            self.sub_vel = self.sub_vel / speed * max_speed

        # Update position
        self.sub_pos += self.sub_vel

        # Screen bounds
        self.sub_pos[0] = np.clip(self.sub_pos[0], 15, self.WIDTH - 15)
        self.sub_pos[1] = np.clip(self.sub_pos[1], 0, self.HEIGHT - 15)
        if self.sub_pos[0] in (15, self.WIDTH - 15):
            self.sub_vel[0] = 0
        if self.sub_pos[1] == self.HEIGHT - 15:
            self.sub_vel[1] = max(0, self.sub_vel[1])


    def _update_entities(self):
        # Update currents
        difficulty_scaling = 1.0 + (self.steps / (10 * self.FPS)) * 0.05
        current_speed = self.CURRENT_BASE_SPEED * difficulty_scaling

        for current in self.currents:
            current["pos"] += current["vel"] * current_speed
            if not (0 < current["pos"][0] < self.WIDTH and 0 < current["pos"][1] < self.HEIGHT):
                current["vel"] = self.np_random.uniform(-1, 1, size=2)
                # Place it back just off-screen to re-enter naturally
                if current["pos"][0] <= 0: current["pos"][0] = -5
                if current["pos"][0] >= self.WIDTH: current["pos"][0] = self.WIDTH + 5
                if current["pos"][1] <= 0: current["pos"][1] = -5
                if current["pos"][1] >= self.HEIGHT: current["pos"][1] = self.HEIGHT + 5


        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - 0.05)

        # Update speed boost
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1
    
    def _spawn_particles(self, angle_offset):
        if self.steps % 3 == 0: # Limit particle spawn rate
            # sfx: bubbles
            base_angle = math.atan2(self.sub_vel[1], self.sub_vel[0]) + angle_offset
            for _ in range(3):
                angle = base_angle + self.np_random.uniform(-0.5, 0.5)
                speed = self.np_random.uniform(0.5, 1.5)
                self.particles.append({
                    "pos": self.sub_pos.copy() + np.array([math.cos(angle+math.pi)*15, math.sin(angle+math.pi)*5]),
                    "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                    "life": self.np_random.integers(20, 41),
                    "radius": self.np_random.uniform(2, 4),
                    "color": (200, 220, 255, self.np_random.integers(50, 151))
                })

    def _handle_collisions(self):
        reward = 0
        # Sub vs Currents
        in_current = False
        for current in self.currents:
            if np.linalg.norm(self.sub_pos - current["pos"]) < current["radius"]:
                self.buoyancy += self.CURRENT_FORCE
                in_current = True
        if in_current:
            # sfx: water current rumble
            pass

        # Sub vs Orbs
        orbs_to_remove = []
        for i, orb_pos in enumerate(self.orbs):
            if np.linalg.norm(self.sub_pos - orb_pos) < 20: # player radius + orb radius
                orbs_to_remove.append(i)
                self.orbs_collected += 1
                self.speed_boost_timer = self.ORB_SPEED_BOOST_DURATION
                reward += 5
                # sfx: orb collection
        
        if orbs_to_remove:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in orbs_to_remove]
            if not self.orbs: # Respawn orbs if all are collected
                 self.orbs = [
                    np.array([
                        self.np_random.uniform(50, self.WIDTH - 50),
                        self.np_random.uniform(50, self.HEIGHT - 80),
                    ]) for _ in range(self.NUM_ORBS)
                ]

        return reward

    def _calculate_reward(self):
        reward = 0
        
        # Reward for vertical movement
        y_change = self.last_y - self.sub_pos[1]
        if y_change > 0:
            reward += 0.1 # Moving up
        elif y_change < 0:
            reward -= 0.1 # Moving down
        self.last_y = self.sub_pos[1]

        # Reward for maintaining buoyancy
        current_buoyancy_dist = abs(self.buoyancy - 60.0)
        if current_buoyancy_dist < self.last_buoyancy_dist:
            reward += 1.0 # Moving towards ideal
        elif current_buoyancy_dist > self.last_buoyancy_dist:
            reward -= 1.0 # Moving away from ideal
        self.last_buoyancy_dist = current_buoyancy_dist

        return reward

    def _check_termination(self):
        # Win condition
        if self.sub_pos[1] <= 10:
            return True
        # Loss conditions
        if not (5 <= self.buoyancy <= 95): # Wider range to allow recovery
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_entities()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            # Interpolate between top and bottom colors
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Surface line
        pygame.draw.rect(self.screen, self.COLOR_SURFACE, (0, 0, self.WIDTH, 10))

    def _render_entities(self):
        # Render Currents
        for current in self.currents:
            pos = current["pos"].astype(int)
            radius = int(current["radius"])
            anim_phase = (self.steps / self.FPS * 2 + current["anim_offset"])
            # Draw multiple transparent circles for a swirling effect
            for i in range(4):
                r = int(radius * (0.6 + 0.4 * math.sin(anim_phase + i * 1.5)))
                offset_x = int(10 * math.cos(anim_phase * 0.5 + i))
                offset_y = int(10 * math.sin(anim_phase * 0.5 + i))
                pygame.gfxdraw.filled_circle(self.screen, pos[0] + offset_x, pos[1] + offset_y, r, self.COLOR_CURRENT)

        # Render Orbs
        for orb_pos in self.orbs:
            pos = orb_pos.astype(int)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, self.COLOR_ORB_GLOW)
            # Solid orb
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, self.COLOR_ORB)

        # Render Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), p["color"])

        # Render Player
        pos_int = self.sub_pos.astype(int)
        angle = math.radians(-self.sub_vel[0] * self.PLAYER_TILT_FACTOR)
        
        # Submarine body shape (a rotated ellipse)
        player_surf = pygame.Surface((60, 40), pygame.SRCALPHA)
        player_rect = player_surf.get_rect(center=(30, 20))
        
        # Glow
        pygame.draw.ellipse(player_surf, self.COLOR_PLAYER_GLOW, player_rect.inflate(10, 10))
        
        # Body
        pygame.draw.ellipse(player_surf, self.COLOR_PLAYER, player_rect)
        
        # Cockpit
        pygame.draw.circle(player_surf, (100, 200, 255), (player_rect.centerx + 5, player_rect.centery), 8)
        
        rotated_surf = pygame.transform.rotate(player_surf, math.degrees(angle))
        rotated_rect = rotated_surf.get_rect(center=pos_int)
        
        self.screen.blit(rotated_surf, rotated_rect.topleft)

    def _render_ui(self):
        # --- UI Panel ---
        ui_y = 10
        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (15, ui_y))
        
        # Depth
        depth = (self.HEIGHT - self.sub_pos[1]) * 2 # Arbitrary depth scale
        depth_text = self.font_large.render(f"DEPTH: {depth:.0f}m", True, self.COLOR_TEXT)
        self.screen.blit(depth_text, (180, ui_y))

        # Orbs
        orb_text = self.font_large.render(f"ORBS: {self.orbs_collected}", True, self.COLOR_TEXT)
        self.screen.blit(orb_text, (350, ui_y))

        # --- Buoyancy Meter ---
        bar_x, bar_y, bar_w, bar_h = self.WIDTH - 40, 80, 20, self.HEIGHT - 160
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        
        # Safe zone indicator
        safe_y = bar_y + bar_h * (1 - 70/100)
        safe_h = bar_h * (20/100)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_GOOD, (bar_x, safe_y, bar_w, safe_h), 2)
        
        # Buoyancy level indicator
        level_h = bar_h * (self.buoyancy / 100)
        level_y = bar_y + bar_h - level_h
        color = self.COLOR_UI_BAR_GOOD if 50 <= self.buoyancy <= 70 else self.COLOR_UI_BAR_WARN if 40 < self.buoyancy < 80 else self.COLOR_UI_BAR_BAD
        pygame.draw.rect(self.screen, color, (bar_x, level_y, bar_w, level_h))
        
        buoyancy_text = self.font_small.render("BUOYANCY", True, self.COLOR_TEXT)
        self.screen.blit(buoyancy_text, (self.WIDTH - 70, bar_y - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "buoyancy": self.buoyancy,
            "depth": self.sub_pos[1],
            "orbs_collected": self.orbs_collected
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Example ---
    # To run this, you must unset the dummy video driver
    # and ensure you have a display environment.
    # For example, run with:
    # SDL_VIDEODRIVER=x11 python your_script_name.py
    
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Disabling dummy video driver for manual play...")
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for manual control
    pygame.display.init()
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Buoyant Submarine")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("\n" + GameEnv.game_description)
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("----------------\n")
    
    running = True
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()