import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:52:19.539870
# Source Brief: brief_00663.md
# Brief Index: 663
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A real-time strategy, wave defense game where the player commands cloned units
    to defend dimensional anchor points against geometric invaders. The agent controls
    a deployment cursor, selects unit types, and places them on the battlefield.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your dimensional anchors from waves of geometric invaders by deploying various "
        "cloned units in this real-time strategy game."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the deployment cursor. Press 'shift' to cycle "
        "through available unit types and 'space' to deploy a unit."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2500
    MAX_WAVES = 20
    FPS = 30

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_ANCHOR = (0, 200, 100)
    COLOR_ANCHOR_GLOW = (0, 200, 100, 50)
    COLOR_INVADER = (255, 50, 50)
    COLOR_INVADER_GLOW = (255, 50, 50, 70)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_PARTICLE_TELEPORT = (255, 255, 0)
    COLOR_PARTICLE_EXPLOSION = (255, 150, 50)
    
    # Unit Type Definitions
    UNIT_SPECS = {
        0: {"name": "Blaster", "cost": 25, "hp": 50, "dmg": 5, "range": 100, "fire_rate": 20, "color": (50, 150, 255)},
        1: {"name": "Gatling", "cost": 40, "hp": 40, "dmg": 2, "range": 80, "fire_rate": 5, "color": (100, 200, 255)},
        2: {"name": "Cannon", "cost": 75, "hp": 100, "dmg": 25, "range": 150, "fire_rate": 60, "color": (0, 100, 200)},
        3: {"name": "Guardian", "cost": 100, "hp": 200, "dmg": 10, "range": 120, "fire_rate": 30, "color": (150, 150, 255)},
    }
    
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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode
        
        # State variables are initialized in reset() to ensure clean episodes
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = np.array([0.0, 0.0])
        self.anchors = []
        self.units = []
        self.invaders = []
        self.projectiles = []
        self.particles = []
        self.wave_number = 0
        self.resources = 0
        self.unlocked_units = 1
        self.selected_unit_type = 0
        self.invaders_to_spawn = 0
        self.wave_cooldown = 0
        self.space_was_held = False
        self.shift_was_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        
        anchor_positions = [
            (self.WIDTH * 0.2, self.HEIGHT / 2),
            (self.WIDTH * 0.5, self.HEIGHT / 2),
            (self.WIDTH * 0.8, self.HEIGHT / 2)
        ]
        self.anchors = [{"pos": np.array(p), "hp": 100, "max_hp": 100} for p in anchor_positions]
        
        self.units = []
        self.invaders = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.resources = 80
        self.unlocked_units = 1
        self.selected_unit_type = 0
        
        self.space_was_held = False
        self.shift_was_held = False
        
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        reward += self._update_units()
        reward += self._update_projectiles()
        reward += self._update_invaders()
        self._update_particles()
        
        wave_reward = self._update_wave_state()
        reward += wave_reward

        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.wave_number > self.MAX_WAVES:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
            self.score += reward
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Logic ---
    
    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        cursor_speed = 5.0
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        # Cycle unit type on shift press (rising edge detection)
        if shift_held and not self.shift_was_held:
            self.selected_unit_type = (self.selected_unit_type + 1) % self.unlocked_units
            # SFX: UI_Cycle.wav
        self.shift_was_held = shift_held
        
        # Deploy unit on space press (rising edge detection)
        if space_held and not self.space_was_held:
            self._deploy_unit()
        self.space_was_held = space_held
        
    def _deploy_unit(self):
        spec = self.UNIT_SPECS[self.selected_unit_type]
        if self.resources >= spec["cost"]:
            self.resources -= spec["cost"]
            self.units.append({
                "pos": self.cursor_pos.copy(),
                "type": self.selected_unit_type,
                "hp": spec["hp"],
                "cooldown": 0,
            })
            self._create_particles(self.cursor_pos, 20, self.COLOR_PARTICLE_TELEPORT, 2.0)
            # SFX: Unit_Deploy.wav
        else:
            # SFX: Error.wav
            pass

    def _update_units(self):
        for unit in self.units:
            unit["cooldown"] = max(0, unit["cooldown"] - 1)
            if unit["cooldown"] == 0 and self.invaders:
                spec = self.UNIT_SPECS[unit["type"]]
                # Find closest invader in range
                target = None
                min_dist = float('inf')
                for invader in self.invaders:
                    dist = np.linalg.norm(unit["pos"] - invader["pos"])
                    if dist < spec["range"] and dist < min_dist:
                        min_dist = dist
                        target = invader
                
                if target:
                    self.projectiles.append({
                        "pos": unit["pos"].copy(),
                        "target_pos": target["pos"].copy(),
                        "speed": 10.0,
                        "dmg": spec["dmg"],
                        "color": spec["color"]
                    })
                    unit["cooldown"] = spec["fire_rate"]
                    # SFX: Unit_Fire_Blaster.wav or Unit_Fire_Gatling.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            direction = self.normalize(p["target_pos"] - p["pos"])
            p["pos"] += direction * p["speed"]
            
            hit = False
            for invader in self.invaders:
                if np.linalg.norm(p["pos"] - invader["pos"]) < 10:
                    invader["hp"] -= p["dmg"]
                    hit = True
                    self._create_particles(p["pos"], 5, self.COLOR_PARTICLE_EXPLOSION, 1.0)
                    if invader["hp"] <= 0:
                        reward += 0.1
                        self.resources += 5 # Gain resources for kill
                        self.invaders.remove(invader)
                        self._create_particles(p["pos"], 15, self.COLOR_PARTICLE_EXPLOSION, 3.0)
                        # SFX: Invader_Explode.wav
                    break
            
            if not hit and 0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT:
                projectiles_to_keep.append(p)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_invaders(self):
        reward = 0
        for invader in self.invaders:
            target_anchor = self.anchors[invader["target_idx"]]
            if np.linalg.norm(invader["pos"] - target_anchor["pos"]) < 10:
                damage = invader["dmg"]
                target_anchor["hp"] -= damage
                reward -= 0.5 * damage
                self.invaders.remove(invader)
                self._create_particles(target_anchor["pos"], 10, self.COLOR_INVADER, 2.5)
                # SFX: Anchor_Damage.wav
            else:
                direction = self.normalize(target_anchor["pos"] - invader["pos"])
                invader["pos"] += direction * invader["speed"]
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _update_wave_state(self):
        reward = 0
        if not self.invaders and self.invaders_to_spawn == 0:
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else:
                if self.wave_number > 0: # Don't give reward for wave 0 -> 1
                    reward += 2.0
                    # SFX: Wave_Complete.wav
                self._start_new_wave()
        
        if self.invaders_to_spawn > 0 and self.steps % 30 == 0: # Spawn one invader per second
            self._spawn_invader()
            self.invaders_to_spawn -= 1
        
        return reward

    def _start_new_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        # Unlock units every 3 waves (after waves 3, 6, 9)
        if self.wave_number > 3 and self.unlocked_units < 2: self.unlocked_units = 2
        if self.wave_number > 6 and self.unlocked_units < 3: self.unlocked_units = 3
        if self.wave_number > 9 and self.unlocked_units < 4: self.unlocked_units = 4
        
        self.invaders_to_spawn = 2 + self.wave_number
        self.wave_cooldown = 90 # 3 second pause between waves

    def _spawn_invader(self):
        # Spawn at a random edge
        edge = self.np_random.integers(4)
        if edge == 0: pos = [0, self.np_random.uniform(0, self.HEIGHT)] # Left
        elif edge == 1: pos = [self.WIDTH, self.np_random.uniform(0, self.HEIGHT)] # Right
        elif edge == 2: pos = [self.np_random.uniform(0, self.WIDTH), 0] # Top
        else: pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT] # Bottom

        base_hp = 10
        base_speed = 0.8
        base_dmg = 5
        
        scaling_factor = 1 + (0.05 * (self.wave_number - 1))
        
        self.invaders.append({
            "pos": np.array(pos, dtype=float),
            "hp": base_hp * scaling_factor,
            "max_hp": base_hp * scaling_factor,
            "speed": base_speed * scaling_factor,
            "dmg": base_dmg,
            "target_idx": self.np_random.integers(len(self.anchors))
        })

    def _check_termination(self):
        if self.wave_number > self.MAX_WAVES:
            return True
        if all(anchor["hp"] <= 0 for anchor in self.anchors):
            return True
        return False

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_game(self):
        for anchor in self.anchors:
            self._render_anchor(anchor)
        for unit in self.units:
            self._render_unit(unit)
        for invader in self.invaders:
            self._render_invader(invader)
        for p in self.projectiles:
            self._render_projectile(p)
        for p in self.particles:
            self._render_particle(p)
        self._render_cursor()

    def _render_anchor(self, anchor):
        pos = (int(anchor["pos"][0]), int(anchor["pos"][1]))
        radius = 20
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, self.COLOR_ANCHOR_GLOW)
        # Main body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ANCHOR)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ANCHOR)
        # Health bar
        if anchor["hp"] < anchor["max_hp"]:
            bar_w = 40
            bar_h = 5
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - radius - 15
            health_ratio = max(0, anchor["hp"] / anchor["max_hp"])
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ANCHOR, (bar_x, bar_y, int(bar_w * health_ratio), bar_h))

    def _render_unit(self, unit):
        pos = (int(unit["pos"][0]), int(unit["pos"][1]))
        spec = self.UNIT_SPECS[unit["type"]]
        size = 10
        # Range indicator
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(spec["range"]), (*spec["color"], 30))
        # Main body
        pygame.draw.rect(self.screen, spec["color"], (pos[0] - size//2, pos[1] - size//2, size, size))
        # Cooldown indicator
        if unit["cooldown"] > 0:
            angle = (unit["cooldown"] / spec["fire_rate"]) * 360
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.arc(self.screen, (255, 255, 255, 100), rect, math.radians(90), math.radians(90 + angle), 2)

    def _render_invader(self, invader):
        pos = invader["pos"]
        direction = self.normalize(self.anchors[invader["target_idx"]]["pos"] - pos)
        angle = math.atan2(direction[1], direction[0])
        
        size = 8
        p1 = (pos[0] + math.cos(angle) * size, pos[1] + math.sin(angle) * size)
        p2 = (pos[0] + math.cos(angle + 2.2) * size, pos[1] + math.sin(angle + 2.2) * size)
        p3 = (pos[0] + math.cos(angle - 2.2) * size, pos[1] + math.sin(angle - 2.2) * size)
        points = [p1, p2, p3]
        int_points = [(int(p[0]), int(p[1])) for p in points]

        # Main body
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_INVADER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_INVADER)

    def _render_projectile(self, p):
        pos = (int(p["pos"][0]), int(p["pos"][1]))
        pygame.draw.circle(self.screen, p["color"], pos, 3)

    def _render_particle(self, p):
        size = int(p["life"] * p["size_ratio"])
        if size > 0:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    def _render_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        radius = 15
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0] - radius, pos[1]), (pos[0] + radius, pos[1]))
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1] - radius), (pos[0], pos[1] + radius))

    def _render_ui(self):
        # Top bar
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        res_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(res_text, (10, 30))
        self.screen.blit(score_text, (10, 50))
        
        # Bottom bar - Unit Selection
        bar_height = 60
        bar_y = self.HEIGHT - bar_height
        pygame.draw.rect(self.screen, (0,0,0,150), (0, bar_y, self.WIDTH, bar_height))
        
        for i in range(self.unlocked_units):
            spec = self.UNIT_SPECS[i]
            box_x = 20 + i * 140
            box_y = bar_y + 10
            box_w, box_h = 120, 40
            
            is_selected = (i == self.selected_unit_type)
            can_afford = self.resources >= spec["cost"]
            
            border_color = self.COLOR_CURSOR if is_selected else (100, 100, 100)
            pygame.draw.rect(self.screen, border_color, (box_x, box_y, box_w, box_h), 2 if is_selected else 1)
            
            text_color = self.COLOR_TEXT if can_afford else (150, 50, 50)
            name_text = self.font_small.render(spec["name"], True, text_color)
            cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, text_color)
            self.screen.blit(name_text, (box_x + 5, box_y + 5))
            self.screen.blit(cost_text, (box_x + 5, box_y + 20))
        
        # Game Over Text
        if self.game_over:
            outcome_text = "VICTORY" if self.wave_number > self.MAX_WAVES else "DEFEAT"
            text_surf = self.font_large.render(outcome_text, True, self.COLOR_CURSOR)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    # --- Utility Methods ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "anchor_health": [a["hp"] for a in self.anchors]
        }
        
    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 30),
                "color": color,
                "size_ratio": self.np_random.uniform(0.1, 0.2)
            })
            
    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else np.array([0.0, 0.0])

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    
    # --- Human player controls ---
    # Arrows: Move cursor
    # Shift: Cycle unit type
    # Space: Deploy unit
    
    # The following code is for human interaction and visualization.
    # It is not part of the Gymnasium environment.
    # To run this, you might need to unset the SDL_VIDEODRIVER dummy variable.
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass
        
    pygame.display.set_caption("Cloned Dimensions Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            obs, info = env.reset()

        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()