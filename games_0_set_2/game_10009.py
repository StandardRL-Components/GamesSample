import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a visually-rich arcade racing game.
    The player pilots a ship through a quasar field, absorbing energy,
    terraforming pathways, and shooting colored energy bursts to gain speed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a ship through a quasar field, absorbing energy, terraforming pathways, and shooting colored energy bursts to gain speed."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire a laser and shift to create a pathway."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_STARS = (200, 200, 220)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_LASER = (255, 255, 255)
    COLOR_PATHWAY = (220, 220, 255)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_ACCENT = (0, 255, 255)
    COLOR_RED = (255, 50, 50)
    COLOR_GREEN = (50, 255, 50)
    COLOR_BLUE = (50, 50, 255)

    # Player settings
    PLAYER_START_SIZE = 12.0
    PLAYER_MIN_SIZE = 6.0
    PLAYER_MAX_SIZE = 24.0
    PLAYER_ACCELERATION = 0.8
    PLAYER_FRICTION = 0.95
    PLAYER_MAX_SPEED = 10.0
    PLAYER_PATHWAY_SPEED_BOOST = 1.3
    PLAYER_SIZE_DECAY_RATE = 0.005 # passive size loss per step

    # Game rules
    TARGET_SPEED = 15.0
    MAX_EPISODE_STEPS = 5000
    LASER_COOLDOWN_STEPS = 5
    LASER_SPEED = 15.0
    TERRAFORM_COOLDOWN_STEPS = 10
    TERRAFORM_SIZE_COST = 0.5
    ENERGY_BURST_SPAWN_CHANCE = 0.05
    MAX_ENERGY_BURSTS = 15
    QUASAR_BASE_FLUCTUATION_SPEED = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 48)

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.last_move_direction = None
        self.laser_cooldown = None
        self.terraform_cooldown = None
        self.quasar_radius = None
        self.quasar_fluctuation_speed = None
        self.stars = None
        self.energy_bursts = None
        self.lasers = None
        self.pathways = None
        self.particles = None
        self.rgb_sequence = None
        self.current_sequence_index = None
        self.steps = None
        self.score = None
        self.game_over = None

        # self.reset() # This is called by the test harness, no need to call it here.
        # self.validate_implementation() # This is a debug tool, not needed for the final env.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = self.PLAYER_START_SIZE
        self.last_move_direction = pygame.Vector2(0, -1) # Default up

        # Cooldowns
        self.laser_cooldown = 0
        self.terraform_cooldown = 0

        # World state
        self.quasar_radius = self.SCREEN_HEIGHT / 3
        self.quasar_fluctuation_speed = self.QUASAR_BASE_FLUCTUATION_SPEED
        self.stars = [
            (
                self.np_random.integers(0, self.SCREEN_WIDTH),
                self.np_random.integers(0, self.SCREEN_HEIGHT),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(150)
        ]
        self.energy_bursts = []
        self.lasers = []
        self.pathways = []
        self.particles = []

        # Gameplay mechanics state
        self.rgb_sequence = [self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE]
        self.current_sequence_index = 0

        # Gym state
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward_this_step = 0

        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        reward_this_step += self._handle_actions(space_held, shift_held)

        # --- 2. Update Game Logic ---
        self._update_player()
        self._update_lasers()
        reward_this_step += self._update_energy_bursts()
        self._update_particles()
        self._update_world()

        # --- 3. Calculate Rewards & Check Termination ---
        # Continuous penalty for being small
        if self.player_size < self.PLAYER_START_SIZE:
            reward_this_step -= 0.01

        # Passive size decay
        self.player_size = max(self.PLAYER_MIN_SIZE, self.player_size - self.PLAYER_SIZE_DECAY_RATE)

        terminated = False
        if self.player_size <= self.PLAYER_MIN_SIZE:
            reward_this_step -= 100
            terminated = True
            self._create_particle_burst(self.player_pos, self.COLOR_RED, 100, 5)
            # Sound: Player_Death_Explosion.wav
        
        player_speed = self.player_vel.length()
        if player_speed >= self.TARGET_SPEED:
            reward_this_step += 100
            terminated = True
            self._create_particle_burst(self.player_pos, self.COLOR_UI_ACCENT, 100, 5)
            # Sound: Victory_Fanfare.wav

        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated should also be true in most cases

        self.game_over = terminated or truncated
        self.score += reward_this_step

        return (
            self._get_observation(),
            reward_this_step,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_movement(self, movement_action):
        move_vec = pygame.Vector2(0, 0)
        if movement_action == 1: # Up
            move_vec.y = -1
        elif movement_action == 2: # Down
            move_vec.y = 1
        elif movement_action == 3: # Left
            move_vec.x = -1
        elif movement_action == 4: # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            self.player_vel += move_vec.normalize() * self.PLAYER_ACCELERATION
            self.last_move_direction = move_vec.normalize()

    def _handle_actions(self, space_held, shift_held):
        reward = 0
        # Fire laser
        if space_held and self.laser_cooldown <= 0:
            self.laser_cooldown = self.LASER_COOLDOWN_STEPS
            laser_pos = self.player_pos + self.last_move_direction * (self.player_size + 5)
            laser_vel = self.last_move_direction * self.LASER_SPEED
            self.lasers.append({"pos": laser_pos, "vel": laser_vel})
            # Sound: Laser_Fire.wav
            self._create_particle_burst(self.player_pos + self.last_move_direction * self.player_size, self.COLOR_PLAYER, 3, 1.0)


        # Terraform pathway
        if shift_held and self.terraform_cooldown <= 0 and self.player_size > self.PLAYER_MIN_SIZE + self.TERRAFORM_SIZE_COST:
            self.terraform_cooldown = self.TERRAFORM_COOLDOWN_STEPS
            self.player_size -= self.TERRAFORM_SIZE_COST
            self.pathways.append(pygame.Vector2(self.player_pos))
            if len(self.pathways) > 100: # Limit path memory
                self.pathways.pop(0)
            # Sound: Terraform_Activate.wav
            self._create_particle_burst(self.player_pos, self.COLOR_PATHWAY, 10, 1.5)

        return reward

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION

        # Apply pathway speed boost
        on_pathway = False
        if len(self.pathways) > 1:
            for p1, p2 in zip(self.pathways, self.pathways[1:]):
                if self._point_segment_distance(self.player_pos, p1, p2) < self.player_size:
                    on_pathway = True
                    break
        
        current_max_speed = self.PLAYER_MAX_SPEED * (self.PLAYER_PATHWAY_SPEED_BOOST if on_pathway else 1.0)
        
        # Clamp speed
        if self.player_vel.length() > current_max_speed:
            self.player_vel.scale_to_length(current_max_speed)

        # Update position and wrap around screen
        self.player_pos += self.player_vel
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT
        
        # Clamp size
        self.player_size = np.clip(self.player_size, self.PLAYER_MIN_SIZE, self.PLAYER_MAX_SIZE)

        # Update cooldowns
        if self.laser_cooldown > 0: self.laser_cooldown -= 1
        if self.terraform_cooldown > 0: self.terraform_cooldown -= 1

    def _update_lasers(self):
        for laser in self.lasers[:]:
            laser["pos"] += laser["vel"]
            if not self.screen.get_rect().collidepoint(laser["pos"]):
                self.lasers.remove(laser)

    def _update_energy_bursts(self):
        reward = 0
        # Spawn new bursts
        if len(self.energy_bursts) < self.MAX_ENERGY_BURSTS and self.np_random.random() < self.ENERGY_BURST_SPAWN_CHANCE:
            pos = pygame.Vector2(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
            color_idx = self.np_random.integers(0, len(self.rgb_sequence))
            color = self.rgb_sequence[color_idx]
            self.energy_bursts.append({
                "pos": pos,
                "color": color,
                "radius": 10,
                "pulse_phase": self.np_random.random() * math.pi * 2
            })

        # Check for collisions
        for burst in self.energy_bursts[:]:
            # Player collision (absorption)
            if self.player_pos.distance_to(burst["pos"]) < self.player_size + burst["radius"]:
                self.player_size += 1.0
                reward += 0.1
                self.energy_bursts.remove(burst)
                self._create_particle_burst(burst["pos"], self.COLOR_PLAYER, 20, 2)
                # Sound: Absorb_Energy.wav
                continue

            # Laser collision
            for laser in self.lasers[:]:
                if laser["pos"].distance_to(burst["pos"]) < burst["radius"]:
                    target_color = self.rgb_sequence[self.current_sequence_index]
                    if burst["color"] == target_color:
                        reward += 1.0
                        self.current_sequence_index += 1
                        # Sound: Sequence_Correct.wav
                        if self.current_sequence_index >= len(self.rgb_sequence):
                            reward += 5.0
                            self.current_sequence_index = 0
                            self.player_vel *= 1.5 # Speed boost for completing sequence
                            self._create_particle_burst(burst["pos"], (255,255,0), 80, 4)
                            # Sound: Sequence_Complete.wav
                        else:
                            self._create_particle_burst(burst["pos"], burst["color"], 40, 3)
                    else: # Incorrect hit
                        reward -= 0.5 # Penalty for wrong hit
                        self.current_sequence_index = 0
                        self._create_particle_burst(burst["pos"], (100,100,100), 20, 2)
                        # Sound: Sequence_Fail.wav
                    
                    self.energy_bursts.remove(burst)
                    self.lasers.remove(laser)
                    break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_world(self):
        # Update quasar fluctuation speed
        if self.steps > 0 and self.steps % 200 == 0:
            self.quasar_fluctuation_speed = min(0.5, self.quasar_fluctuation_speed + 0.05)
        
        # Animate quasar radius
        self.quasar_radius = (self.SCREEN_HEIGHT / 3) + math.sin(self.steps * self.quasar_fluctuation_speed) * (self.SCREEN_HEIGHT / 6)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            brightness = 180 + math.sin(pygame.time.get_ticks() * 0.001 * size) * 40
            star_color = (brightness, brightness, brightness + 20)
            pygame.gfxdraw.pixel(self.screen, int(x), int(y), star_color)

        # Quasar
        quasar_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        quasar_color_val = 50 + 30 * math.sin(self.steps * self.quasar_fluctuation_speed * 0.5)
        quasar_color = (int(quasar_color_val/2), int(quasar_color_val/3), int(quasar_color_val))
        pygame.gfxdraw.filled_circle(self.screen, quasar_center[0], quasar_center[1], int(self.quasar_radius), quasar_color)
        pygame.gfxdraw.aacircle(self.screen, quasar_center[0], quasar_center[1], int(self.quasar_radius), quasar_color)

        # Pathways
        if len(self.pathways) > 1:
            glow_path = [(int(p.x), int(p.y)) for p in self.pathways]
            pygame.draw.lines(self.screen, self.COLOR_PATHWAY, False, glow_path, 3)
        
        # Energy Bursts
        for burst in self.energy_bursts:
            pulse = 1 + 0.2 * math.sin(self.steps * 0.1 + burst["pulse_phase"])
            radius = int(burst["radius"] * pulse)
            pos = (int(burst["pos"].x), int(burst["pos"].y))
            color_glow = tuple(min(255, int(c*0.5)) for c in burst["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 2, color_glow)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, burst["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, burst["color"])

        # Lasers
        for laser in self.lasers:
            start_pos = (int(laser["pos"].x), int(laser["pos"].y))
            end_pos = (int(laser["pos"].x - laser["vel"].x), int(laser["pos"].y - laser["vel"].y))
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            color = p["color"] + (alpha,)
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        size_int = int(self.player_size)
        # Glow
        glow_radius = int(size_int * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW + (80,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_pos_int[0] - glow_radius, player_pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Ship body
        angle = self.last_move_direction.angle_to(pygame.Vector2(0, -1))
        points = [
            pygame.Vector2(0, -size_int).rotate(-angle) + self.player_pos,
            pygame.Vector2(-size_int * 0.7, size_int * 0.7).rotate(-angle) + self.player_pos,
            pygame.Vector2(size_int * 0.7, size_int * 0.7).rotate(-angle) + self.player_pos,
        ]
        points_int = [(int(p.x), int(p.y)) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, points_int, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points_int, self.COLOR_PLAYER)

    def _render_ui(self):
        # Speed display
        speed = self.player_vel.length()
        speed_text = self.font_ui.render(f"SPEED: {speed:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (10, 10))

        # Target speed display
        target_text = self.font_ui.render(f"TARGET: {self.TARGET_SPEED:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_text, (self.SCREEN_WIDTH - target_text.get_width() - 10, 10))

        # Ship size bar
        bar_width = 150
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 15
        size_ratio = np.clip((self.player_size - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE), 0, 1)
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (bar_x, bar_y, bar_width * size_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # RGB sequence display
        seq_y = self.SCREEN_HEIGHT - 30
        for i, color in enumerate(self.rgb_sequence):
            box_size = 20
            box_x = (self.SCREEN_WIDTH // 2) - (len(self.rgb_sequence) * (box_size + 5) // 2) + i * (box_size + 5)
            rect = (box_x, seq_y, box_size, box_size)
            if i < self.current_sequence_index:
                pygame.draw.rect(self.screen, color, rect)
            else:
                pygame.draw.rect(self.screen, (50,50,50), rect)
                if i == self.current_sequence_index:
                    pygame.draw.rect(self.screen, color, rect, 3) # Highlight current target
                else:
                    pygame.draw.rect(self.screen, (80,80,80), rect, 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player_size,
            "player_speed": self.player_vel.length()
        }

    def _create_particle_burst(self, position, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(position),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "max_life": 30,
                "color": color,
                "size": self.np_random.integers(1, 4)
            })

    def _point_segment_distance(self, p, a, b):
        # Using https://stackoverflow.com/a/1501725
        if a == b: return p.distance_to(a)
        l2 = a.distance_squared_to(b)
        if l2 == 0.0: return p.distance_to(a)
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return p.distance_to(projection)
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in the test environment, but is useful for local testing.
    # It requires the "dummy" video driver to be unset.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen for display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Racer")

    total_reward = 0
    while not done:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        rendered_frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(env.screen, rendered_frame)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(GameEnv.TARGET_FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()