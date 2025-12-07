import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:36:37.392052
# Source Brief: brief_00584.md
# Brief Index: 584
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Survive waves of malfunctioning machines by strategically launching size-changing
    wrenches enhanced with elemental properties.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of malfunctioning machines by strategically launching size-changing "
        "wrenches enhanced with elemental properties."
    )
    user_guide = (
        "Controls: ↑/↓ to adjust launch power. ←/→ to select wrench type. "
        "Hold space to charge your wrench, release to fire."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 12000  # ~4 minutes at 50 FPS target
    TARGET_FPS = 50

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_STATION = (100, 220, 255)
    COLOR_STATION_GLOW = (50, 110, 130)
    COLOR_HEALTH_BAR = (40, 255, 120)
    COLOR_HEALTH_BAR_BG = (255, 60, 60)
    COLOR_TEXT = (220, 230, 240)
    COLOR_MACHINE_BODY = (255, 80, 80)
    COLOR_MACHINE_SPARK = (255, 220, 50)
    COLOR_WRENCH = (200, 200, 210)
    COLOR_WRENCH_TRAIL = (180, 180, 190)

    CARD_COLORS = {
        "standard": (200, 200, 210),
        "electric": (0, 180, 255),
        "magnetic": (255, 255, 0),
        "piercing": (200, 100, 255)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.station_health = 0
        self.station_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 40)
        self.station_radius = 25
        self.machines = []
        self.wrenches = []
        self.particles = []
        
        # Player control state
        self.launch_power = 0.0
        self.launch_angle_deg = 0.0
        self.wrench_charge = 0.0
        self.was_space_held = False
        self.last_move_direction = 1 # Start pointing up
        
        # Card system
        self.card_deck = ["standard", "electric", "magnetic", "piercing"]
        self.selected_card_idx = 0
        
        # Difficulty scaling
        self.machine_spawn_timer = 0
        self.machine_spawn_rate = 150 # Spawn every 150 steps initially
        self.machine_speed = 1.0
        self.machine_health = 1

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.station_health = 100
        self.machines.clear()
        self.wrenches.clear()
        self.particles.clear()
        
        self.launch_power = 5.0
        self.launch_angle_deg = -90.0
        self.wrench_charge = 0.0
        self.was_space_held = False
        self.last_move_direction = 1 # Up
        self.selected_card_idx = 0
        
        self.machine_spawn_timer = 0
        self.machine_spawn_rate = 150
        self.machine_speed = 1.0
        self.machine_health = 1

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # 1. Process player input from action
        movement, space_held_int, shift_held_int = action
        space_held = space_held_int == 1
        shift_held = shift_held_int == 1
        
        # Aiming and Power (Movement)
        if movement != 0:
            self.last_move_direction = movement
        if movement == 1: self.launch_power = min(10.0, self.launch_power + 0.2) # Up
        if movement == 2: self.launch_power = max(2.0, self.launch_power - 0.2) # Down
        if movement == 3: self.selected_card_idx = (self.selected_card_idx - 1) % len(self.card_deck) # Left
        if movement == 4: self.selected_card_idx = (self.selected_card_idx + 1) % len(self.card_deck) # Right

        # Update angle based on last direction
        angle_map = {1: -90, 2: 90, 3: -135, 4: -45} # Up, Down, Left, Right
        self.launch_angle_deg = angle_map.get(self.last_move_direction, self.launch_angle_deg)
        
        # Wrench Charging (Space)
        if space_held:
            self.wrench_charge = min(1.0, self.wrench_charge + 0.05)
        
        # Launch Wrench (Space Release)
        if not space_held and self.was_space_held:
            self._launch_wrench()
            self.wrench_charge = 0.0
            reward -= 0.1 # Small cost for launching
        self.was_space_held = space_held

        # Angle Adjustment (Shift) - This is unused as per brief adaptation
        # if shift_held: ...

        # 2. Update game logic
        self._update_difficulty()
        self._spawn_machines()
        self._update_wrenches()
        collision_reward = self._update_machines()
        self._update_particles()
        reward += collision_reward

        # 3. Check for termination
        terminated = self.station_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.station_health > 0:
                reward += 100 # Survival bonus
            else:
                reward -= 100 # Penalty for station destruction
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _launch_wrench(self):
        # sfx: wrench_launch.wav
        size = 10 + 20 * self.wrench_charge
        angle_rad = math.radians(self.launch_angle_deg)
        velocity_x = self.launch_power * math.cos(angle_rad)
        velocity_y = self.launch_power * math.sin(angle_rad)
        card_type = self.card_deck[self.selected_card_idx]
        
        self.wrenches.append({
            "pos": list(self.station_pos),
            "vel": [velocity_x, velocity_y],
            "angle": 0,
            "rot_speed": 15 * (1 - self.wrench_charge),
            "size": size,
            "type": card_type,
            "pierce_count": 3 if card_type == "piercing" else 1,
            "trail": []
        })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 600 == 0:
            self.machine_spawn_rate = max(30, self.machine_spawn_rate - 15)
            self.machine_speed = min(3.0, self.machine_speed + 0.2)
            if self.steps % 1200 == 0:
                self.machine_health += 1

    def _spawn_machines(self):
        self.machine_spawn_timer += 1
        if self.machine_spawn_timer >= self.machine_spawn_rate:
            self.machine_spawn_timer = 0
            side = random.choice([0, 1, 2]) # 0: top, 1: left, 2: right
            if side == 0:
                x = random.uniform(50, self.SCREEN_WIDTH - 50)
                y = -20
            elif side == 1:
                x = -20
                y = random.uniform(50, self.SCREEN_HEIGHT - 150)
            else: # side == 2
                x = self.SCREEN_WIDTH + 20
                y = random.uniform(50, self.SCREEN_HEIGHT - 150)
            
            self.machines.append({
                "pos": [x, y],
                "radius": random.uniform(15, 25),
                "health": self.machine_health,
                "max_health": self.machine_health
            })

    def _update_wrenches(self):
        for w in self.wrenches[:]:
            w["pos"][0] += w["vel"][0]
            w["vel"][1] += 0.1  # Gravity
            w["pos"][1] += w["vel"][1]
            w["angle"] += w["rot_speed"]
            
            w["trail"].append(tuple(w["pos"]))
            if len(w["trail"]) > 10:
                w["trail"].pop(0)

            if not (0 < w["pos"][0] < self.SCREEN_WIDTH and -50 < w["pos"][1] < self.SCREEN_HEIGHT):
                self.wrenches.remove(w)

    def _update_machines(self):
        collision_reward = 0
        for m in self.machines[:]:
            # Move towards station
            dx = self.station_pos[0] - m["pos"][0]
            dy = self.station_pos[1] - m["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist > 1:
                m["pos"][0] += (dx / dist) * self.machine_speed
                m["pos"][1] += (dy / dist) * self.machine_speed

            # Check collision with station
            if dist < self.station_radius + m["radius"]:
                # sfx: station_hit.wav
                damage = 10
                self.station_health -= damage
                collision_reward -= 5
                self._create_particles(m["pos"], 30, self.COLOR_MACHINE_BODY)
                self.machines.remove(m)
                continue
            
            # Check collision with wrenches
            for w in self.wrenches:
                if w["pierce_count"] <= 0: continue
                wrench_dist = math.hypot(w["pos"][0] - m["pos"][0], w["pos"][1] - m["pos"][1])
                if wrench_dist < w["size"] / 2 + m["radius"]:
                    # sfx: wrench_hit.wav
                    m["health"] -= 1
                    collision_reward += 0.5
                    self._create_particles(w["pos"], 15, self.CARD_COLORS[w["type"]])
                    
                    w["pierce_count"] -= 1
                    if w["pierce_count"] <= 0 and w["type"] != "piercing":
                        if w in self.wrenches: self.wrenches.remove(w)

                    if m["health"] <= 0:
                        # sfx: machine_repaired.wav
                        collision_reward += 2
                        self.score += 10 * m["max_health"]
                        self._create_particles(m["pos"], 40, self.COLOR_HEALTH_BAR)
                        if m in self.machines: self.machines.remove(m)
                        break
        return collision_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": random.randint(15, 30),
                "color": color,
                "radius": random.uniform(1, 3)
            })

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_machines()
        self._render_station()
        self._render_wrenches()
        self._render_ui()

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = p["color"]
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]),
                int(p["radius"]), (*color, alpha)
            )

    def _render_machines(self):
        for m in self.machines:
            pos = (int(m["pos"][0]), int(m["pos"][1]))
            radius = int(m["radius"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_MACHINE_BODY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_MACHINE_BODY)
            
            # Spark effect
            spark_angle = (self.steps * 10 + id(m)) % 360 * (math.pi/180)
            spark_pos = (
                int(pos[0] + math.cos(spark_angle) * radius * 0.7),
                int(pos[1] + math.sin(spark_angle) * radius * 0.7)
            )
            pygame.draw.circle(self.screen, self.COLOR_MACHINE_SPARK, spark_pos, 3)
            
            # Health bar
            if m["health"] < m["max_health"]:
                health_pct = m["health"] / m["max_health"]
                bar_width = radius * 1.5
                bar_height = 5
                bar_x = pos[0] - bar_width / 2
                bar_y = pos[1] - radius - 10
                pygame.draw.rect(self.screen, (100,0,0), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_station(self):
        pos = (int(self.station_pos[0]), int(self.station_pos[1]))
        radius = int(self.station_radius)
        # Glow
        glow_radius = int(radius * 1.5 + abs(math.sin(self.steps * 0.05)) * 5)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_STATION_GLOW, 100))
        # Base
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_STATION)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_STATION)
        
        # Health Bar
        bar_width = 150
        bar_height = 15
        bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
        bar_y = self.SCREEN_HEIGHT - 20
        health_pct = max(0, self.station_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=4)

    def _render_wrenches(self):
        for w in self.wrenches:
            # Trail
            for i, p in enumerate(w["trail"]):
                alpha = int(255 * (i / len(w["trail"])))
                color = (*self.CARD_COLORS[w["type"]], alpha)
                pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), int(w["size"] * 0.2 * (i/len(w["trail"]))))

            # Wrench body
            size = w["size"]
            pos = (int(w["pos"][0]), int(w["pos"][1]))
            angle_rad = math.radians(w["angle"])
            
            # Simple wrench shape using a rotated rectangle
            points = [(-size/2, -size/4), (size/2, -size/4), (size/2, size/4), (-size/2, size/4)]
            rotated_points = []
            for x, y in points:
                rx = pos[0] + x * math.cos(angle_rad) - y * math.sin(angle_rad)
                ry = pos[1] + x * math.sin(angle_rad) + y * math.cos(angle_rad)
                rotated_points.append((rx, ry))
            
            pygame.draw.polygon(self.screen, self.CARD_COLORS[w["type"]], rotated_points)
            pygame.draw.aalines(self.screen, (255,255,255), True, rotated_points)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Aiming / Power indicator
        aim_end_x = self.station_pos[0] + self.launch_power * 10 * math.cos(math.radians(self.launch_angle_deg))
        aim_end_y = self.station_pos[1] + self.launch_power * 10 * math.sin(math.radians(self.launch_angle_deg))
        pygame.draw.line(self.screen, (255, 255, 255, 100), self.station_pos, (aim_end_x, aim_end_y), 2)
        
        # Wrench charge indicator
        if self.wrench_charge > 0:
            charge_angle = -math.pi * self.wrench_charge
            pygame.draw.arc(self.screen, self.COLOR_WRENCH, 
                            (self.station_pos[0]-40, self.station_pos[1]-40, 80, 80),
                            -math.pi, charge_angle, 5)

        # Card selection
        card_size = 40
        card_padding = 10
        total_width = len(self.card_deck) * (card_size + card_padding) - card_padding
        start_x = self.SCREEN_WIDTH / 2 - total_width / 2
        for i, card_type in enumerate(self.card_deck):
            x = start_x + i * (card_size + card_padding)
            y = self.SCREEN_HEIGHT - 80
            rect = pygame.Rect(x, y, card_size, card_size)
            color = self.CARD_COLORS[card_type]
            
            if i == self.selected_card_idx:
                pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(8, 8), border_radius=8)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=6)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.station_health > 0 else "REPAIR FAILED"
            end_text = self.font_title.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            score_text_end = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_text_end.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(score_text_end, score_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # Example usage: Play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Wrench Repair Mayhem")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Manual Control ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.TARGET_FPS)

    env.close()