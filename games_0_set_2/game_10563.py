import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Generated: 2025-08-26T14:40:59.094070
# Source Brief: brief_00563.md
# Brief Index: 563
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a charged Hadron.
    The goal is to navigate a procedurally generated obstacle course,
    collecting charges to maintain speed and stability.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a charged Hadron through a perilous obstacle course. "
        "Collect charges to maintain speed and stability while avoiding collisions."
    )
    user_guide = (
        "Controls: Use ↑↓ to aim your launch, then press space to start. "
        "Hold space to boost and press shift to regenerate stability."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors - Vibrant sci-fi theme
        self.COLOR_BG = (10, 10, 26) # Deep space blue
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_HADRON = (255, 34, 85)
        self.COLOR_HADRON_GLOW = (255, 100, 125)
        self.COLOR_HURDLE = (34, 255, 136)
        self.COLOR_HURDLE_GLOW = (100, 255, 175)
        self.COLOR_SPEED_CHARGE = (68, 170, 255)
        self.COLOR_STABILITY_CHARGE = (255, 255, 102)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_UI_STABILITY = (34, 200, 136)
        self.COLOR_UI_STABILITY_LOW = (255, 100, 100)
        
        # Game state variables
        self.level = None
        self.skill_points = None
        self.boost_level = None
        self.regen_level = None
        self.hadron_pos = None
        self.hadron_vel = None
        self.hadron_stability = None
        self.launch_angle = None
        self.game_state = None
        self.regen_cooldown = None
        self.particles = None
        self.hurdles = None
        self.charges = None
        self.steps = None
        self.score = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Hard reset for a new "game", soft reset for a new "level"
        hard_reset = options.get("hard_reset", False) if options else False
        if self.level is None or hard_reset:
            self.level = 1
            self.skill_points = 0
            self.boost_level = 1.0
            self.regen_level = 1.0

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_state = "LAUNCH_AIM"
        
        # Hadron state
        self.hadron_pos = pygame.Vector2(60, self.HEIGHT / 2)
        self.hadron_vel = pygame.Vector2(0, 0)
        self.hadron_stability = 100.0
        self.launch_angle = -math.pi / 6 # Default aim
        
        # Cooldowns and entity lists
        self.regen_cooldown = 0
        self.particles = []
        self.hurdles = []
        self.charges = []

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        truncated = False

        self.steps += 1

        if self.game_state == "LAUNCH_AIM":
            # --- AIMING PHASE ---
            if movement == 1: # Up
                self.launch_angle -= 0.05
            elif movement == 2: # Down
                self.launch_angle += 0.05
            self.launch_angle = np.clip(self.launch_angle, -math.pi / 2, math.pi / 2)

            if space_held:
                # Launch the hadron
                launch_power = 10
                self.hadron_vel = pygame.Vector2(
                    math.cos(self.launch_angle) * launch_power,
                    math.sin(self.launch_angle) * launch_power
                )
                self.game_state = "PLAYING"
                # Sound: Player Launch
        
        elif self.game_state == "PLAYING":
            # --- PLAYING PHASE ---
            reward += 0.1 # Survival reward

            # Stability drain
            self.hadron_stability -= 0.05 
            
            # Action: Speed Boost (Space)
            if space_held:
                boost_cost = 0.5
                if self.hadron_stability > boost_cost:
                    boost_power = 0.5 * self.boost_level
                    self.hadron_vel *= (1 + boost_power / self.hadron_vel.length()) if self.hadron_vel.length() > 0 else 1
                    self.hadron_stability -= boost_cost
                    # Visual: Create boost particles
                    for _ in range(3):
                        self._create_particle(self.hadron_pos, self.COLOR_HADRON, 1.5, -self.hadron_vel)
                    # Sound: Boost activate
            
            # Action: Stability Regen (Shift)
            if shift_held and self.regen_cooldown == 0:
                self.hadron_stability = min(100.0, self.hadron_stability + 25 * self.regen_level)
                self.regen_cooldown = 180 # 3 second cooldown at 60fps
                # Visual: Create regen pulse effect
                self.particles.append({"pos": self.hadron_pos.copy(), "type": "pulse", "radius": 10, "max_radius": 50, "life": 20})
                # Sound: Regen activate

            # Update hadron position
            self.hadron_pos += self.hadron_vel

            # Add gravity
            self.hadron_vel.y += 0.1

            # Collision checks
            collision_reward, terminated = self._check_collisions()
            reward += collision_reward

        # Update systems
        self._update_particles()
        if self.regen_cooldown > 0:
            self.regen_cooldown -= 1

        # Check termination conditions
        if self.hadron_stability <= 0:
            reward = -100.0
            terminated = True
            # Sound: Player fail
        
        if self.hadron_pos.x > self.WIDTH:
            # Course complete
            reward = 100.0
            terminated = True
            self.level += 1
            self.skill_points += 1
            # Auto-upgrade skills
            if self.skill_points > 0:
                if self.boost_level <= self.regen_level:
                    self.boost_level += 0.1
                else:
                    self.regen_level += 0.1
                self.skill_points -= 1
            # Sound: Level complete
        
        if self.steps >= 2000:
            truncated = True
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        hurdle_count = 5 + int(self.level * 1.0) # Difficulty scaling
        charge_count = 3 + int(self.level * 0.5)

        for _ in range(hurdle_count):
            while True:
                x = self.np_random.integers(200, self.WIDTH - 100)
                y = self.np_random.integers(50, self.HEIGHT - 50)
                w = self.np_random.integers(15, 25 + 1)
                h = self.np_random.integers(60, 120 + 1)
                hurdle = pygame.Rect(x, y, w, h)
                # Ensure no hurdle at start
                if not hurdle.colliderect(pygame.Rect(40, self.HEIGHT/2 - 50, 40, 100)):
                    self.hurdles.append(hurdle)
                    break

        for _ in range(charge_count):
            x = self.np_random.integers(150, self.WIDTH - 80)
            y = self.np_random.integers(50, self.HEIGHT - 50)
            charge_type = self.np_random.choice(["speed", "stability"])
            self.charges.append({"pos": pygame.Vector2(x, y), "type": charge_type, "pulse": self.np_random.uniform(0, math.pi * 2)})

    def _check_collisions(self):
        hadron_rect = pygame.Rect(self.hadron_pos.x - 8, self.hadron_pos.y - 8, 16, 16)

        # Hurdles
        for hurdle in self.hurdles:
            if hurdle.colliderect(hadron_rect):
                # Sound: Explosion
                for _ in range(50):
                    self._create_particle(self.hadron_pos, self.COLOR_HURDLE, 3.0)
                return -100.0, True

        # Charges
        reward = 0.0
        for charge in self.charges[:]:
            charge_pos = charge["pos"]
            if (charge_pos - self.hadron_pos).length() < 20: # 8 (hadron) + 12 (charge)
                if charge["type"] == "speed":
                    self.hadron_vel *= 1.1
                    reward += 1.0
                    color = self.COLOR_SPEED_CHARGE
                    # Sound: Speed charge collect
                else: # stability
                    self.hadron_stability = min(100.0, self.hadron_stability + 15)
                    reward += 0.5
                    color = self.COLOR_STABILITY_CHARGE
                    # Sound: Stability charge collect
                
                for _ in range(20):
                    self._create_particle(charge_pos, color, 2.0)
                self.charges.remove(charge)
        
        # Boundaries
        if not (0 < self.hadron_pos.y < self.HEIGHT):
            # Sound: Wall bounce
            self.hadron_vel.y *= -0.8 # Bouncy walls
            self.hadron_pos.y = np.clip(self.hadron_pos.y, 1, self.HEIGHT - 1)
            self.hadron_stability -= 5

        return reward, False

    def _create_particle(self, pos, color, speed_mult=1.0, base_vel=None):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3) * speed_mult
        vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        if base_vel:
            vel += base_vel * 0.3
        self.particles.append({
            "pos": pos.copy(),
            "vel": vel,
            "life": self.np_random.integers(20, 40 + 1),
            "color": color,
            "size": self.np_random.uniform(2, 5)
        })

    def _update_particles(self):
        for p in self.particles[:]:
            if p.get("type") == "pulse":
                p["radius"] += (p["max_radius"] - p["radius"]) * 0.2
                p["life"] -= 1
                if p["life"] <= 0:
                    self.particles.remove(p)
            else:
                p["pos"] += p["vel"]
                p["life"] -= 1
                p["size"] *= 0.95
                if p["life"] <= 0 or p["size"] < 0.5:
                    self.particles.remove(p)

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
            "level": self.level,
            "stability": self.hadron_stability,
            "game_state": self.game_state,
        }

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw particles
        for p in self.particles:
            if p.get("type") == "pulse":
                alpha = int(255 * (p["life"] / 20))
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), self.COLOR_STABILITY_CHARGE + (alpha,))
            else:
                pos = (int(p["pos"].x), int(p["pos"].y))
                pygame.draw.circle(self.screen, p["color"], pos, int(p["size"]))

        # Draw hurdles
        for hurdle in self.hurdles:
            pygame.draw.rect(self.screen, self.COLOR_HURDLE, hurdle)
            pygame.draw.rect(self.screen, self.COLOR_HURDLE_GLOW, hurdle.inflate(4,4), 1)

        # Draw charges
        for charge in self.charges:
            pos = (int(charge["pos"].x), int(charge["pos"].y))
            pulse_size = 3 * math.sin(self.steps * 0.1 + charge["pulse"])
            color = self.COLOR_SPEED_CHARGE if charge["type"] == "speed" else self.COLOR_STABILITY_CHARGE
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(12 + pulse_size), color)

        # Draw hadron trail
        if self.game_state == "PLAYING":
            self._create_particle(self.hadron_pos, self.COLOR_HADRON_GLOW, 0.1, -self.hadron_vel * 0.5)

        # Draw hadron
        if self.hadron_pos:
            pos = (int(self.hadron_pos.x), int(self.hadron_pos.y))
            glow_radius = int(12 + self.hadron_stability / 20)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_HADRON_GLOW + (50,))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_HADRON)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_HADRON)
        
        # Draw launch trajectory
        if self.game_state == "LAUNCH_AIM":
            start_pos = self.hadron_pos
            end_pos = start_pos + pygame.Vector2(math.cos(self.launch_angle), math.sin(self.launch_angle)) * 80
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, start_pos, end_pos, 2)
            # Draw "AIMING" text
            text_surf = self.font_large.render("AIM & LAUNCH", True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, (self.WIDTH/2 - text_surf.get_width()/2, self.HEIGHT/2 - text_surf.get_height()/2))
            
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Level
        level_text = self.font_main.render(f"COURSE: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        # Stability bar
        stability_perc = max(0, self.hadron_stability / 100.0)
        bar_width = 200
        bar_color = self.COLOR_UI_STABILITY if stability_perc > 0.3 else self.COLOR_UI_STABILITY_LOW
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.WIDTH/2 - bar_width/2 - 2, 8, bar_width + 4, 24))
        pygame.draw.rect(self.screen, bar_color, (self.WIDTH/2 - bar_width/2, 10, bar_width * stability_perc, 20))
        stab_text = self.font_main.render("STABILITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(stab_text, (self.WIDTH/2 - stab_text.get_width()/2, 12))

        # Regen Cooldown
        if self.regen_cooldown > 0:
            cooldown_perc = self.regen_cooldown / 180.0
            text = self.font_main.render(f"REGEN PULSE: {cooldown_perc:.0%}", True, self.COLOR_UI_STABILITY_LOW)
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, 35))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the autograder, but is useful for testing.
    # Make sure to remove "dummy" from SDL_VIDEODRIVER to see the window.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hadron Runner")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Level Reached: {info['level']}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)
        
    env.close()