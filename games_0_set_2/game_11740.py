import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:27:24.514333
# Source Brief: brief_01740.md
# Brief Index: 1740
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Atlantis Survival'.

    The player must survive in the sinking city of Atlantis by gathering energy
    from ruins, teleporting between them, and avoiding catastrophic tidal waves.
    The goal is to survive for 5000 steps.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Switch ruin state
    - actions[2]: Shift button (0=released, 1=held) -> Unused

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive the sinking city of Atlantis. Teleport between ruins to gather energy, "
        "switch their state to avoid tidal waves, and outlast the rising waters."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport between ruins. "
        "Press space to switch a ruin's state between land and sea."
    )
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    MAX_STEPS = 5000

    # --- Visuals & Colors ---
    COLOR_BG = (10, 25, 40)
    COLOR_BG_SHIMMER = (20, 40, 60)
    COLOR_LAND = (255, 200, 80)
    COLOR_LAND_GLOW = (255, 200, 80, 50)
    COLOR_SEA = (40, 100, 180)
    COLOR_SEA_GLOW = (40, 100, 180, 50)
    COLOR_INACTIVE = (30, 40, 50)
    COLOR_PLAYER = (255, 0, 128)
    COLOR_PLAYER_GLOW = (255, 0, 128, 50)
    COLOR_DANGER = (220, 20, 60)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_ENERGY = (0, 255, 255)

    # --- Gameplay Mechanics ---
    TIDE_RISE_INTERVAL = 500
    WAVE_BASE_INTERVAL = 800
    WAVE_FREQ_INCREASE_INTERVAL = 1000
    UNDERWATER_TELEPORT_UNLOCK_STEP = 2500
    ENERGY_PER_STEP_LAND = 0.5
    STATE_SWITCH_COOLDOWN = 15  # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self._create_ruin_definitions()
        # self.reset() is called by the wrapper, no need to call it here.

    def _create_ruin_definitions(self):
        self.ruin_definitions = []
        grid = [(x, y) for y in [100, 200, 300] for x in [120, 320, 520]]
        for i, (x, y) in enumerate(grid):
            neighbors = {
                1: i - 3 if i >= 3 else -1,  # Up
                2: i + 3 if i < 6 else -1,   # Down
                3: i - 1 if i % 3 != 0 else -1, # Left
                4: i + 1 if (i + 1) % 3 != 0 else -1, # Right
            }
            self.ruin_definitions.append({
                "id": i,
                "pos": pygame.Vector2(x, y),
                "radius": 25,
                "activation_step": i * 300,
                "neighbors": neighbors,
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.energy = 50
        self.game_over = False
        
        self.ruins = []
        for definition in self.ruin_definitions:
            ruin = definition.copy()
            ruin["state"] = 'land' if self.np_random.random() > 0.5 else 'sea'
            ruin["is_active"] = ruin["activation_step"] == 0
            ruin["is_destroyed"] = False
            ruin["switch_cooldown"] = 0
            self.ruins.append(ruin)

        self.player_ruin_index = 0
        self.tide_level = self.SCREEN_HEIGHT + 50
        
        self.wave_timer = self.WAVE_BASE_INTERVAL
        self.wave_progress = -1 # -1 means inactive
        self.wave_duration = 120 # steps for wave to cross screen
        self.wave_freq_modifier = 1.0

        self.particles = []
        self.effects = [] # For ripples, etc.
        self.message = ""
        self.message_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Update Game State ---
        self._update_timers_and_cooldowns()
        reward += self._update_environment()
        
        # --- Handle Player Action ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        action_reward = self._handle_player_action(movement, space_held)
        reward += action_reward
        
        # --- Update Energy ---
        player_ruin = self.ruins[self.player_ruin_index]
        if not player_ruin["is_destroyed"]:
            if player_ruin["state"] == 'land':
                self.energy += self.ENERGY_PER_STEP_LAND
                reward += self.ENERGY_PER_STEP_LAND * 2 # Mapped to +1 as per brief
                # Add energy particles
                if self.steps % 5 == 0:
                    self._add_particles(player_ruin["pos"], 1, self.COLOR_ENERGY, 0.5, -2, 20)
            else:
                reward -= 0.1 # Penalty for being in submerged ruin

        self.energy = max(0, self.energy)

        # --- Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        if terminated:
            self.score += terminal_reward
            if self.steps >= self.MAX_STEPS:
                self._set_message("ATLANTIS SURVIVED!", self.COLOR_ENERGY, 300)
            else:
                self._set_message("SYSTEMS FAILED", self.COLOR_DANGER, 300)

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_timers_and_cooldowns(self):
        if self.message_timer > 0:
            self.message_timer -= 1
        for ruin in self.ruins:
            if ruin["switch_cooldown"] > 0:
                ruin["switch_cooldown"] -= 1

    def _update_environment(self):
        reward = 0
        # Update tide
        if self.steps > 0 and self.steps % self.TIDE_RISE_INTERVAL == 0:
            self.tide_level -= 20
        
        # Update ruins based on tide and activation
        for ruin in self.ruins:
            if not ruin["is_destroyed"] and ruin["pos"].y > self.tide_level:
                ruin["is_destroyed"] = True
                self._add_effect(ruin["pos"], "burst", 30, self.COLOR_INACTIVE)
            if not ruin["is_active"] and self.steps >= ruin["activation_step"]:
                ruin["is_active"] = True
                reward += 10
                self._add_effect(ruin["pos"], "burst", 30, self.COLOR_LAND)
                self._set_message("New Ruin Online", self.COLOR_LAND, 60)

        # Unlock underwater teleport
        if self.steps == self.UNDERWATER_TELEPORT_UNLOCK_STEP:
            reward += 10
            self._set_message("Sub-Aquatic Teleport Unlocked", self.COLOR_SEA, 120)

        # Update tidal wave
        if self.wave_progress >= 0:
            self.wave_progress += 1
            if self.wave_progress > self.wave_duration:
                self.wave_progress = -1
                reward += 5 # Survived wave
        else:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_progress = 0
                if self.steps > 0 and self.steps % self.WAVE_FREQ_INCREASE_INTERVAL == 0:
                    self.wave_freq_modifier *= 0.9
                self.wave_timer = int(self.WAVE_BASE_INTERVAL * self.wave_freq_modifier)
        
        return reward

    def _handle_player_action(self, movement, space_held):
        reward = 0
        player_ruin = self.ruins[self.player_ruin_index]

        # Action: Switch State
        if space_held and self.energy >= 10 and player_ruin["switch_cooldown"] == 0 and not player_ruin["is_destroyed"]:
            self.energy -= 10
            player_ruin["state"] = 'sea' if player_ruin["state"] == 'land' else 'land'
            player_ruin["switch_cooldown"] = self.STATE_SWITCH_COOLDOWN
            self._add_effect(player_ruin["pos"], "ripple", 40, 
                             self.COLOR_LAND if player_ruin["state"] == 'land' else self.COLOR_SEA)
        
        # Action: Teleport
        if movement != 0:
            target_idx = player_ruin["neighbors"].get(movement, -1)
            if target_idx != -1:
                target_ruin = self.ruins[target_idx]
                can_teleport = (
                    target_ruin["is_active"] and 
                    not target_ruin["is_destroyed"] and 
                    self.energy >= 5
                )
                # Check underwater teleport unlock
                if target_ruin["state"] == 'sea' and self.steps < self.UNDERWATER_TELEPORT_UNLOCK_STEP:
                    can_teleport = False

                if can_teleport:
                    self.energy -= 5
                    self._add_particles(player_ruin["pos"], 20, self.COLOR_PLAYER, 2, 0, 30)
                    self.player_ruin_index = target_idx
                    self._add_effect(self.ruins[self.player_ruin_index]["pos"], "implode", 30, self.COLOR_PLAYER)
        
        return reward

    def _check_termination(self):
        # Win condition
        if self.steps >= self.MAX_STEPS:
            return True, 100.0

        # Loss condition: Hit by wave
        if self.wave_progress >= 0:
            wave_y = (self.wave_progress / self.wave_duration) * self.SCREEN_HEIGHT
            player_y = self.ruins[self.player_ruin_index]["pos"].y
            if abs(wave_y - player_y) < 10:
                return True, -100.0
        
        # Loss condition: All ruins destroyed
        active_ruins = [r for r in self.ruins if r["is_active"] and not r["is_destroyed"]]
        if not active_ruins:
            return True, -100.0

        # Loss condition: No energy to move
        if self.energy < 5:
            # check if current ruin provides energy
            current_ruin = self.ruins[self.player_ruin_index]
            if current_ruin["state"] == 'sea' or current_ruin["is_destroyed"]:
                # Check if any neighbor is reachable
                can_move = False
                for move_dir in range(1, 5):
                    neighbor_idx = current_ruin["neighbors"].get(move_dir, -1)
                    if neighbor_idx != -1:
                        neighbor_ruin = self.ruins[neighbor_idx]
                        if neighbor_ruin["is_active"] and not neighbor_ruin["is_destroyed"]:
                             can_move = True # even if not enough energy, a path exists
                             break
                if not can_move: # Trapped without energy source
                    return True, -100.0

        return False, 0.0

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "tide_level": self.tide_level,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_ruins()
        self._render_effects()
        self._render_player()
        self._render_wave()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_effects(self):
        # Water shimmer
        for i in range(4):
            offset = (self.steps + i * 100) * 0.02
            amplitude = 3
            y_base = 50 + i * 100
            points = []
            for x in range(0, self.SCREEN_WIDTH + 20, 20):
                y = y_base + math.sin(x * 0.01 + offset) * amplitude
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_BG_SHIMMER, False, points, 1)

        # Rising tide visual
        tide_rect = pygame.Rect(0, self.tide_level, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.tide_level)
        pygame.draw.rect(self.screen, self.COLOR_BG_SHIMMER, tide_rect)

    def _render_ruins(self):
        for ruin in self.ruins:
            pos = (int(ruin["pos"].x), int(ruin["pos"].y))
            radius = int(ruin["radius"])
            
            if ruin["is_destroyed"]:
                color = self.COLOR_INACTIVE
            elif not ruin["is_active"]:
                color = self.COLOR_INACTIVE
            elif ruin["state"] == 'land':
                color = self.COLOR_LAND
            else:
                color = self.COLOR_SEA
            
            # Draw base platform
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

            # Draw glow
            if ruin["is_active"] and not ruin["is_destroyed"]:
                glow_color = self.COLOR_LAND_GLOW if ruin["state"] == 'land' else self.COLOR_SEA_GLOW
                for i in range(3):
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 2 + i*2, glow_color)

    def _render_player(self):
        if self.game_over: return
        ruin = self.ruins[self.player_ruin_index]
        pos = (int(ruin["pos"].x), int(ruin["pos"].y))
        
        # Pulsating glow
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        for i in range(5):
            alpha = int(self.COLOR_PLAYER_GLOW[3] * (1 - i/5) * (0.5 + pulse * 0.5))
            glow_color = (*self.COLOR_PLAYER_GLOW[:3], alpha)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10 + i * 3, glow_color)

        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
    
    def _render_wave(self):
        if self.wave_progress >= 0:
            y = (self.wave_progress / self.wave_duration) * self.SCREEN_HEIGHT
            
            # Draw main wave line with alpha
            s = pygame.Surface((self.SCREEN_WIDTH, 20))
            s.set_alpha(150)
            s.fill(self.COLOR_DANGER)
            self.screen.blit(s, (0, y - 10))

            # Draw brighter core line
            pygame.draw.line(self.screen, (255, 100, 100), (0, y), (self.SCREEN_WIDTH, y), 2)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_small.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))

        # Energy Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 15
        energy_ratio = min(1, self.energy / 100)
        
        pygame.draw.rect(self.screen, self.COLOR_INACTIVE, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (bar_x, bar_y, bar_width * energy_ratio, bar_height))
        energy_text = self.font_small.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (bar_x + bar_width + 10, bar_y))
        
        # Message
        if self.message_timer > 0:
            alpha = min(255, self.message_timer * 5)
            message_surf = self.font_large.render(self.message, True, self.message_color)
            message_surf.set_alpha(alpha)
            pos = message_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(message_surf, pos)

    def _render_effects(self):
        # Update and draw particles
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
                self.screen.blit(s, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))
        self.particles = new_particles

        # Update and draw other effects
        new_effects = []
        for e in self.effects:
            e['life'] -= 1
            if e['life'] > 0:
                new_effects.append(e)
                progress = 1 - (e['life'] / e['max_life'])
                pos = (int(e['pos'].x), int(e['pos'].y))
                if e['type'] == 'ripple':
                    radius = int(e['size'] * progress)
                    alpha = int(255 * (1 - progress))
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*e['color'], alpha))
                elif e['type'] == 'burst':
                     for i in range(8):
                        angle = i * (math.pi / 4) + progress * 2
                        dist = progress * e['size']
                        x = pos[0] + math.cos(angle) * dist
                        y = pos[1] + math.sin(angle) * dist
                        pygame.draw.line(self.screen, e['color'], pos, (int(x), int(y)), 1)
                elif e['type'] == 'implode':
                    radius = int(e['size'] * (1 - progress))
                    alpha = int(255 * (1 - progress))
                    for i in range(3):
                        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + i*2, (*e['color'], alpha))

        self.effects = new_effects

    def _add_particles(self, pos, count, color, speed, gravity, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_val = self.np_random.uniform(0.5, 1.5) * speed
            vel = pygame.Vector2(math.cos(angle) * vel_val, math.sin(angle) * vel_val + gravity)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "life": self.np_random.integers(lifetime // 2, lifetime),
                "max_life": lifetime,
                "size": self.np_random.integers(1, 4)
            })

    def _add_effect(self, pos, type, size, color):
        self.effects.append({
            "pos": pos.copy(),
            "type": type,
            "size": size,
            "color": color,
            "life": size,
            "max_life": size
        })

    def _set_message(self, text, color, duration):
        self.message = text
        self.message_color = color
        self.message_timer = duration

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block is for manual play and demonstration.
    # It will open a window and let you control the agent.
    # The environment itself is headless as required.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Atlantis Survival - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.TARGET_FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            action.fill(0)
            pygame.time.wait(2000)

    env.close()