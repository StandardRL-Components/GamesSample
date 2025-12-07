import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:37:11.632948
# Source Brief: brief_01114.md
# Brief Index: 1114
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "In a cyberpunk world, deploy various programs to capture rogue glitches. "
        "Collect resources to unlock new districts and more powerful program types."
    )
    user_guide = (
        "Controls: Use ↑/↓ arrows to aim and ←/→ to adjust power. "
        "Press space to deploy a program and shift to cycle between available programs."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (5, 10, 20)
    COLOR_GRID = (20, 40, 80)
    COLOR_AIMER = (255, 255, 255)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (10, 20, 40)
    COLOR_GLITCH = (255, 50, 50)
    COLOR_RESOURCE = (255, 220, 0)
    
    PROGRAM_TYPES = {
        "Seeker": {"color": (0, 255, 255), "speed_mult": 1.0, "size": 5, "unlock_cost": 0},
        "Dart": {"color": (100, 255, 100), "speed_mult": 1.5, "size": 4, "unlock_cost": 250},
        "Breaker": {"color": (255, 100, 255), "speed_mult": 0.8, "size": 8, "unlock_cost": 1000},
    }
    DISTRICT_UNLOCK_THRESHOLDS = [0, 100, 500, 1500, 5000]

    MAX_STEPS = 5000
    FPS = 30
    
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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Persistent state across episodes
        self.total_resources = 0
        self.unlocked_districts = {0}
        self.unlocked_programs = {"Seeker"}
        self.program_keys = list(self.PROGRAM_TYPES.keys())

        # Initialize episode-specific state variables
        self._initialize_episode_state()

    def _initialize_episode_state(self):
        self.steps = 0
        self.score = 0
        self.terminated = False

        self.programs = []
        self.glitches = []
        self.particles = []

        self.aim_angle = -math.pi / 2  # Straight up
        self.aim_power = 50.0  # 0-100
        
        self.deployment_cooldown = 0
        self.glitch_spawn_timer = 0
        
        self.last_space_state = 0
        self.last_shift_state = 0
        
        self.selected_program_idx = 0
        self.available_programs = [p for p in self.program_keys if p in self.unlocked_programs]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_episode_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0
        
        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Deploy Program ---
        program_deployed = self._deploy_program(space_held)
        if program_deployed:
            # Placeholder for sound effect
            # sfx.play('deploy')
            pass

        # --- Cycle Program ---
        cycled, new_program_unlocked = self._cycle_program(shift_held)
        if cycled:
            # sfx.play('cycle_weapon')
            pass

        # --- Update Game State ---
        self._update_programs()
        self._update_glitches()
        self._update_particles()
        
        # --- Spawn New Glitches ---
        self._spawn_glitches()

        # --- Handle Collisions & Collect Resources ---
        glitches_captured = self._handle_collisions()
        if glitches_captured > 0:
            step_reward += 0.1 * glitches_captured
            self.total_resources += glitches_captured
            # sfx.play('capture_glitch')
        
        # --- Check for Progression Unlocks ---
        district_unlocked = self._check_district_unlock()
        if district_unlocked:
            step_reward += 1.0
            # sfx.play('unlock_district')
        
        program_unlocked = self._check_program_unlock()
        if program_unlocked:
            step_reward += 0.5
            self.available_programs = [p for p in self.program_keys if p in self.unlocked_programs]
            # sfx.play('unlock_program')

        self.score += step_reward
        self.steps += 1
        self.deployment_cooldown = max(0, self.deployment_cooldown - 1)
        
        self.terminated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            step_reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Adjust Angle (Up/Down)
        if movement == 1: self.aim_angle -= 0.05
        if movement == 2: self.aim_angle += 0.05
        self.aim_angle = np.clip(self.aim_angle, -math.pi, 0)
        
        # Adjust Power (Left/Right)
        if movement == 3: self.aim_power -= 1.5
        if movement == 4: self.aim_power += 1.5
        self.aim_power = np.clip(self.aim_power, 10, 100)

    def _deploy_program(self, space_held):
        if space_held and not self.last_space_state and self.deployment_cooldown == 0:
            self.last_space_state = True
            self.deployment_cooldown = 10 # 1/3 of a second cooldown
            
            prog_type_name = self.available_programs[self.selected_program_idx]
            prog_type = self.PROGRAM_TYPES[prog_type_name]
            
            power_normalized = self.aim_power / 100.0
            speed = 3 + 7 * power_normalized * prog_type["speed_mult"]
            
            self.programs.append({
                "pos": np.array([self.WIDTH / 2, self.HEIGHT - 20], dtype=float),
                "vel": np.array([math.cos(self.aim_angle) * speed, math.sin(self.aim_angle) * speed]),
                "type": prog_type,
                "trail": deque(maxlen=15)
            })
            return True
        if not space_held:
            self.last_space_state = False
        return False

    def _cycle_program(self, shift_held):
        cycled = False
        newly_unlocked = False
        if shift_held and not self.last_shift_state:
            self.last_shift_state = True
            self.selected_program_idx = (self.selected_program_idx + 1) % len(self.available_programs)
            cycled = True
        if not shift_held:
            self.last_shift_state = False
        return cycled, newly_unlocked

    def _update_programs(self):
        for prog in self.programs:
            prog["trail"].append(prog["pos"].copy())
            prog["pos"] += prog["vel"]
        self.programs = [p for p in self.programs if 0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT]

    def _update_glitches(self):
        for glitch in self.glitches:
            glitch["pos"] += glitch["vel"]
            if not (0 < glitch["pos"][0] < self.WIDTH and 0 < glitch["pos"][1] < self.HEIGHT):
                glitch["vel"] = self._get_new_glitch_velocity(glitch["speed"])
                glitch["pos"] = np.clip(glitch["pos"], [0, 0], [self.WIDTH, self.HEIGHT])

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_current_district_level(self):
        return len(self.unlocked_districts) - 1

    def _spawn_glitches(self):
        level = self._get_current_district_level()
        spawn_rate = max(10, 150 - level * 20) # Ticks between spawns
        self.glitch_spawn_timer -= 1
        if self.glitch_spawn_timer <= 0:
            self.glitch_spawn_timer = spawn_rate
            
            speed = 1.0 + level * 0.5
            start_edge = self.np_random.integers(4)
            if start_edge == 0: pos = np.array([0, self.np_random.uniform(0, self.HEIGHT)], dtype=float)
            elif start_edge == 1: pos = np.array([self.WIDTH, self.np_random.uniform(0, self.HEIGHT)], dtype=float)
            elif start_edge == 2: pos = np.array([self.np_random.uniform(0, self.WIDTH), 0], dtype=float)
            else: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT], dtype=float)
            
            self.glitches.append({
                "pos": pos,
                "vel": self._get_new_glitch_velocity(speed),
                "speed": speed,
                "size": self.np_random.integers(6, 10)
            })

    def _get_new_glitch_velocity(self, speed):
        angle = self.np_random.uniform(0, 2 * math.pi)
        return np.array([math.cos(angle) * speed, math.sin(angle) * speed])

    def _handle_collisions(self):
        captured_count = 0
        progs_to_remove = set()
        glitches_to_remove = set()

        for i, prog in enumerate(self.programs):
            for j, glitch in enumerate(self.glitches):
                if j in glitches_to_remove: continue
                
                dist = np.linalg.norm(prog["pos"] - glitch["pos"])
                if dist < prog["type"]["size"] + glitch["size"]:
                    progs_to_remove.add(i)
                    glitches_to_remove.add(j)
                    captured_count += 1
                    self._create_explosion(glitch["pos"], self.COLOR_GLITCH)
                    self._create_explosion(glitch["pos"], self.COLOR_RESOURCE, count=10, speed=2)
                    break 
        
        if progs_to_remove:
            self.programs = [p for i, p in enumerate(self.programs) if i not in progs_to_remove]
        if glitches_to_remove:
            self.glitches = [g for i, g in enumerate(self.glitches) if i not in glitches_to_remove]
        
        return captured_count

    def _check_district_unlock(self):
        current_level = self._get_current_district_level()
        if current_level + 1 < len(self.DISTRICT_UNLOCK_THRESHOLDS):
            if self.total_resources >= self.DISTRICT_UNLOCK_THRESHOLDS[current_level + 1]:
                self.unlocked_districts.add(current_level + 1)
                return True
        return False
        
    def _check_program_unlock(self):
        unlocked_something = False
        for name, props in self.PROGRAM_TYPES.items():
            if name not in self.unlocked_programs and self.total_resources >= props["unlock_cost"]:
                self.unlocked_programs.add(name)
                unlocked_something = True
        return unlocked_something

    def _create_explosion(self, pos, color, count=20, speed=4):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * vel_mag, math.sin(angle) * vel_mag]),
                "life": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_glitches()
        self._render_programs()
        self._render_aimer()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_resources": self.total_resources,
            "district_level": self._get_current_district_level()
        }

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_particles(self):
        for p in self.particles:
            size = int(p["size"] * (p["life"] / 30.0))
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], p["pos"].astype(int), size)

    def _render_glitches(self):
        for g in self.glitches:
            pos = g["pos"].astype(int)
            size = g["size"]
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, self.COLOR_GLITCH, rect)
            pygame.gfxdraw.rectangle(self.screen, rect, (255, 150, 150))
            # Jitter effect
            if self.np_random.random() > 0.8:
                offset_x = self.np_random.integers(-2, 3)
                offset_y = self.np_random.integers(-2, 3)
                pygame.draw.rect(self.screen, (255, 255, 255), rect.move(offset_x, offset_y), 1)

    def _render_programs(self):
        for p in self.programs:
            # Trail
            if len(p["trail"]) > 1:
                for i, trail_pos in enumerate(p["trail"]):
                    alpha = int(255 * (i / len(p["trail"])))
                    color = (*p["type"]["color"], alpha)
                    temp_surf = pygame.Surface((p["type"]["size"]*2, p["type"]["size"]*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (p["type"]["size"], p["type"]["size"]), int(p["type"]["size"] * (i / len(p["trail"]))))
                    self.screen.blit(temp_surf, trail_pos - p["type"]["size"])

            # Main projectile
            pos_int = p["pos"].astype(int)
            size = p["type"]["size"]
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size, p["type"]["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size, (255, 255, 255))

    def _render_aimer(self):
        start_pos = np.array([self.WIDTH / 2, self.HEIGHT - 20])
        length = 20 + self.aim_power * 0.8
        end_pos = start_pos + np.array([math.cos(self.aim_angle) * length, math.sin(self.aim_angle) * length])
        
        pygame.draw.line(self.screen, self.COLOR_AIMER, start_pos, end_pos, 2)
        pygame.gfxdraw.filled_circle(self.screen, int(start_pos[0]), int(start_pos[1]), 5, self.COLOR_AIMER)
        pygame.gfxdraw.aacircle(self.screen, int(start_pos[0]), int(start_pos[1]), 5, self.COLOR_AIMER)

    def _render_ui(self):
        # Resources
        self._render_text(f"RESOURCES: {int(self.total_resources)}", (10, 10), self.font_main)
        
        # District
        level = self._get_current_district_level()
        self._render_text(f"DISTRICT: {level}", (self.WIDTH - 150, 10), self.font_main)

        # Program selection
        base_y = self.HEIGHT - 40
        for i, prog_name in enumerate(self.available_programs):
            color = self.PROGRAM_TYPES[prog_name]["color"]
            x_pos = self.WIDTH / 2 - (len(self.available_programs) * 70 / 2) + i * 70
            
            box_rect = pygame.Rect(x_pos, base_y, 60, 25)
            
            if i == self.selected_program_idx:
                pygame.draw.rect(self.screen, color, box_rect, 0, 3)
                self._render_text(prog_name, (x_pos + 5, base_y + 3), self.font_small, color=(0,0,0), shadow=False)
            else:
                pygame.draw.rect(self.screen, color, box_rect, 1, 3)
                self._render_text(prog_name, (x_pos + 5, base_y + 3), self.font_small, color=color, shadow=False)
                
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # The __main__ block is for manual play and is not part of the environment's core logic.
    # It will use the display, overriding the headless setting.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Cyberpunk Glitch Hunter")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(GameEnv.user_guide)
    
    while not done:
        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Resources: {info['total_resources']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)
        
    env.close()
    print("Game Over.")