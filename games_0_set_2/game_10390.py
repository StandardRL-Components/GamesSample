import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a splitting light beam.
    The goal is to absorb enough energy to transform before time runs out.

    **Visual Style:** Abstract, neon, futuristic.
    **Core Loop:** Aim beam, collect orbs, manage power/heat, transform.
    **Victory:** Reach 100 power and transform.
    **Failure:** Timer reaches zero.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a beam of light to collect energy orbs. Absorb enough energy to transform before the timer runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to aim your light beam and collect the white orbs."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_TIME_SECONDS = 45
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG_TOP = (5, 0, 20)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_BEAM_COOL = (0, 191, 255)  # DeepSkyBlue
    COLOR_BEAM_HOT = (255, 69, 0)   # OrangeRed
    COLOR_ORB = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)

    # Game Mechanics
    NUM_INITIAL_ORBS = 10
    ORB_RADIUS = 8
    ORB_POWER_VALUE = 10
    POWER_THRESHOLD_FOR_COOLING = 50.0
    POWER_COOLING_RATE = 2.0  # Power per second
    TRANSFORMATION_POWER = 100.0
    BEAM_BASE_SPEED = 2.0
    BEAM_SPEED_POWER_MULTIPLIER = 0.02
    BEAM_ROTATION_SPEED = math.radians(4.0)
    BEAM_HEAD_RADIUS = 6
    BEAM_SPLIT_ANGLE = math.radians(30)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.game_phase = "pre-transformation"
        self.beams = []
        self.orbs = []
        self.particles = []
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME_SECONDS
        self.game_phase = "pre-transformation"
        
        # Initialize the main beam
        self.beams = [{
            "pos": pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "power": 0.0
        }]
        
        self.orbs = []
        self._spawn_orbs(self.NUM_INITIAL_ORBS)
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        # --- Update Time ---
        self.timer -= 1.0 / self.FPS
        
        # --- Handle Input and Update Game State ---
        self._handle_input(action)
        self._update_beams()
        self._update_particles()
        
        orbs_collected = self._handle_collisions()
        if orbs_collected > 0:
            reward += 1.0 * orbs_collected

        transformed_this_step, transform_reward = self._check_transformation()
        if transformed_this_step:
            reward += transform_reward
            
        self._check_orb_respawn()

        # --- Check for Termination ---
        terminated = False
        if self.game_phase == "post-transformation":
            terminated = True
            reward += 100.0  # Victory bonus
            self.score += 100 # Also add to score for display
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100.0  # Loss penalty
        
        self.game_over = terminated
        
        # Truncated is not used in this environment
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        # Actions 1-4 rotate the beam towards a cardinal direction
        if movement != 0:
            target_angle = {
                1: -math.pi / 2,  # Up
                2: math.pi / 2,   # Down
                3: math.pi,       # Left
                4: 0              # Right
            }[movement]
            
            for beam in self.beams:
                # Normalize angles for shortest path rotation
                current_angle = beam["angle"]
                delta = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
                if abs(delta) < self.BEAM_ROTATION_SPEED:
                    beam["angle"] = target_angle
                else:
                    beam["angle"] += math.copysign(self.BEAM_ROTATION_SPEED, delta)
                beam["angle"] %= (2 * math.pi)

    def _update_beams(self):
        dt = 1.0 / self.FPS
        for beam in self.beams:
            # Update speed based on power
            speed = self.BEAM_BASE_SPEED + beam["power"] * self.BEAM_SPEED_POWER_MULTIPLIER
            
            # Update position
            velocity = pygame.Vector2(math.cos(beam["angle"]), math.sin(beam["angle"])) * speed
            beam["pos"] += velocity
            
            # Screen wrapping
            beam["pos"].x %= self.SCREEN_WIDTH
            beam["pos"].y %= self.SCREEN_HEIGHT
            
            # Cool down if power is high
            if beam["power"] > self.POWER_THRESHOLD_FOR_COOLING:
                beam["power"] -= self.POWER_COOLING_RATE * dt
                beam["power"] = max(self.POWER_THRESHOLD_FOR_COOLING, beam["power"])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['initial_radius'] * (p['life'] / p['initial_life']))

    def _handle_collisions(self):
        collected_count = 0
        orbs_to_remove = []
        for orb_idx, orb_pos in enumerate(self.orbs):
            for beam in self.beams:
                if orb_idx not in orbs_to_remove:
                    distance = beam["pos"].distance_to(orb_pos)
                    if distance < self.BEAM_HEAD_RADIUS + self.ORB_RADIUS:
                        orbs_to_remove.append(orb_idx)
                        beam["power"] = min(self.TRANSFORMATION_POWER, beam["power"] + self.ORB_POWER_VALUE)
                        self.score += self.ORB_POWER_VALUE
                        collected_count += 1
                        self._spawn_particles_at(orb_pos)
                        break # An orb can only be collected once
        
        # Remove collected orbs safely
        for orb_idx in sorted(orbs_to_remove, reverse=True):
            del self.orbs[orb_idx]
            
        return collected_count

    def _check_transformation(self):
        if self.game_phase == "pre-transformation" and self.beams[0]["power"] >= self.TRANSFORMATION_POWER:
            main_beam = self.beams[0]
            main_beam["power"] = 40.0 # Reduce power after split
            
            # Create two new beams
            angle1 = (main_beam["angle"] - self.BEAM_SPLIT_ANGLE) % (2 * math.pi)
            angle2 = (main_beam["angle"] + self.BEAM_SPLIT_ANGLE) % (2 * math.pi)
            
            self.beams.append({
                "pos": main_beam["pos"].copy(), "angle": angle1, "power": 30.0
            })
            self.beams.append({
                "pos": main_beam["pos"].copy(), "angle": angle2, "power": 30.0
            })
            
            self.game_phase = "post-transformation"
            return True, 50.0
        return False, 0.0

    def _check_orb_respawn(self):
        if not self.orbs:
            self._spawn_orbs(self.NUM_INITIAL_ORBS)

    def _spawn_orbs(self, count):
        for _ in range(count):
            self.orbs.append(pygame.Vector2(
                self.np_random.uniform(self.ORB_RADIUS, self.SCREEN_WIDTH - self.ORB_RADIUS),
                self.np_random.uniform(self.ORB_RADIUS, self.SCREEN_HEIGHT - self.ORB_RADIUS)
            ))

    def _spawn_particles_at(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'initial_life': life,
                'initial_radius': self.np_random.uniform(1, 3),
                'color': (200, 200, 255)
            })

    def _get_observation(self):
        self._render_background()
        self._render_orbs()
        self._render_particles()
        self._render_beams()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            t = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - t) + self.COLOR_BG_BOTTOM[0] * t),
                int(self.COLOR_BG_TOP[1] * (1 - t) + self.COLOR_BG_BOTTOM[1] * t),
                int(self.COLOR_BG_TOP[2] * (1 - t) + self.COLOR_BG_BOTTOM[2] * t)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_beams(self):
        for beam in self.beams:
            heat_factor = min(1.0, beam["power"] / self.TRANSFORMATION_POWER)
            beam_color = (
                int(self.COLOR_BEAM_COOL[0] * (1 - heat_factor) + self.COLOR_BEAM_HOT[0] * heat_factor),
                int(self.COLOR_BEAM_COOL[1] * (1 - heat_factor) + self.COLOR_BEAM_HOT[1] * heat_factor),
                int(self.COLOR_BEAM_COOL[2] * (1 - heat_factor) + self.COLOR_BEAM_HOT[2] * heat_factor)
            )
            
            pos_int = (int(beam["pos"].x), int(beam["pos"].y))
            
            # Glow effect
            glow_radius = int(self.BEAM_HEAD_RADIUS * 2.5)
            glow_color = (*beam_color, 60) # RGBA with low alpha
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

            # Beam head
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BEAM_HEAD_RADIUS, beam_color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BEAM_HEAD_RADIUS, beam_color)

    def _render_orbs(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 0.2 + 0.9 # Pulse between 90% and 110%
        for orb_pos in self.orbs:
            pos_int = (int(orb_pos.x), int(orb_pos.y))
            radius = int(self.ORB_RADIUS * pulse)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ORB)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ORB)

    def _render_particles(self):
        for p in self.particles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                alpha = int(255 * (p['life'] / p['initial_life']))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos_int[0]-radius, pos_int[1]-radius))

    def _render_ui(self):
        # Power display
        power_text = f"POWER: {int(self.beams[0]['power'])}"
        power_surf = self.font_ui.render(power_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(power_surf, (10, 10))

        # Timer display
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))

        # Game Over messages
        if self.game_over:
            if self.game_phase == "post-transformation":
                msg = "VICTORY"
            else:
                msg = "TIME UP"
            msg_surf = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "power": self.beams[0]['power'] if self.beams else 0,
            "phase": self.game_phase
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Light Beam Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_reward = 0.0
            print("--- ENV RESET ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # In a real scenario, you'd reset here. For manual play, we just show the final screen.
            # obs, info = env.reset()
            # total_reward = 0.0

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()