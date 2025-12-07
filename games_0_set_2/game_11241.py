import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:43:39.745727
# Source Brief: brief_01241.md
# Brief Index: 1241
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Stabilize a volatile anomaly by activating a sequence of nodes while defending the system core from incoming attacks."
    )
    user_guide = (
        "Use the arrow keys to move the selector. Press space to activate a node. Press shift to deploy a temporary shield."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: GYMNASIUM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # --- VISUAL STYLE & CONSTANTS ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_TEXT = (200, 220, 255)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_ATTACK = (255, 50, 50)
        self.COLOR_SHIELD = (100, 150, 255)
        self.NODE_COLORS = {
            "blue": (0, 150, 255),
            "green": (0, 255, 150),
            "purple": (200, 100, 255),
            "gray": (100, 100, 100)
        }
        self.COLOR_NAMES = list(self.NODE_COLORS.keys())[:-1] # Exclude gray

        # --- GAMEPLAY CONSTANTS ---
        self.MAX_STEPS = 1000
        self.SELECTOR_SPEED = 10.0
        self.SHIELD_DURATION = 30 # steps
        self.SHIELD_COOLDOWN = 90 # steps
        
        # --- PERSISTENT STATE ---
        self.difficulty_level = 0
        self.total_wins = 0
        
        # --- INITIALIZE STATE (will be overwritten by reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.instability = 0.0
        self.selector_pos = pygame.math.Vector2(0, 0)
        self.selected_node_idx = None
        self.nodes = []
        self.solution_sequence = []
        self.current_solution_step = 0
        self.puzzle_solved = False
        self.last_space_held = False
        self.last_shift_held = False
        self.shield_active_timer = 0
        self.shield_cooldown_timer = 0
        self.attacks = []
        self.particles = []
        self.active_chains = []
        self.background_stars = []

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.instability = 0.0
        self.selector_pos = pygame.math.Vector2(self.screen_width / 2, self.screen_height / 2)
        self.selected_node_idx = None
        self.puzzle_solved = False
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.shield_active_timer = 0
        self.shield_cooldown_timer = 0
        
        self.attacks = []
        self.particles = []
        self.active_chains = []
        self._generate_puzzle()
        self.background_stars = [
            (random.randint(0, self.screen_width), random.randint(0, self.screen_height), random.randint(1, 2))
            for _ in range(100)
        ]

        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        num_nodes = 5 + 2 * self.difficulty_level
        solution_length = 3 + min(self.difficulty_level, 2)
        
        self.nodes = []
        self.solution_sequence = [random.choice(self.COLOR_NAMES) for _ in range(solution_length)]
        self.current_solution_step = 0
        
        required_colors = {color: self.solution_sequence.count(color) for color in self.COLOR_NAMES}
        
        node_positions = []
        min_dist = 75
        for i in range(num_nodes):
            while True:
                pos = pygame.math.Vector2(
                    random.uniform(50, self.screen_width - 50),
                    random.uniform(50, self.screen_height - 50)
                )
                if all(pos.distance_to(p) > min_dist for p in node_positions):
                    node_positions.append(pos)
                    break

            if any(required_colors.values()):
                color_name = random.choice([c for c, n in required_colors.items() if n > 0])
                required_colors[color_name] -= 1
            else:
                color_name = "gray"

            self.nodes.append({
                "pos": pos,
                "color_name": color_name,
                "color": self.NODE_COLORS[color_name],
                "state": "idle", # idle, active, solved
                "connections": [],
                "radius": 15,
                "activation_progress": 0.0
            })

        # Create visual connections
        for i in range(num_nodes):
            num_connections = random.randint(1, 2 + self.difficulty_level // 2)
            for _ in range(num_connections):
                other_idx = random.randint(0, num_nodes - 1)
                if i != other_idx and other_idx not in self.nodes[i]["connections"]:
                    self.nodes[i]["connections"].append(other_idx)

    def step(self, action):
        reward = 0.0
        
        # --- 1. HANDLE INPUT ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        reward += self._handle_input(movement, space_press, shift_press)
        
        # --- 2. UPDATE GAME LOGIC ---
        reward += self._update_game_logic()
        
        # --- 3. CHECK FOR TERMINATION ---
        self.game_over = self._check_termination()
        
        # --- 4. CALCULATE TERMINAL REWARDS ---
        if self.game_over:
            if self.puzzle_solved:
                reward += 100.0
                self.total_wins += 1
                if self.total_wins > 0 and self.total_wins % 5 == 0:
                    self.difficulty_level = min(5, self.difficulty_level + 1)
            else: # Loss from instability or timeout
                reward += -100.0
        
        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False, # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        step_reward = 0.0
        # Move selector
        if movement == 1: self.selector_pos.y -= self.SELECTOR_SPEED
        elif movement == 2: self.selector_pos.y += self.SELECTOR_SPEED
        elif movement == 3: self.selector_pos.x -= self.SELECTOR_SPEED
        elif movement == 4: self.selector_pos.x += self.SELECTOR_SPEED
        self.selector_pos.x = np.clip(self.selector_pos.x, 0, self.screen_width)
        self.selector_pos.y = np.clip(self.selector_pos.y, 0, self.screen_height)

        # Find closest node to selector
        min_dist = float('inf')
        self.selected_node_idx = None
        for i, node in enumerate(self.nodes):
            dist = self.selector_pos.distance_to(node["pos"])
            if dist < 50: # Selection radius
                min_dist = dist
                self.selected_node_idx = i

        # Activate shield
        if shift_press and self.shield_cooldown_timer <= 0:
            # SFX: Shield_Activate
            self.shield_active_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            self._create_particles(pygame.math.Vector2(self.screen_width/2, self.screen_height/2), 100, self.COLOR_SHIELD, 20)

        # Activate node
        if space_press and self.selected_node_idx is not None:
            node = self.nodes[self.selected_node_idx]
            if node["state"] == "idle":
                # SFX: Node_Activate_Attempt
                correct_color = self.solution_sequence[self.current_solution_step]
                if node["color_name"] == correct_color:
                    # SFX: Correct_Node
                    step_reward += 1.0
                    node["state"] = "solved"
                    self._create_particles(node["pos"], 50, node["color"], 15)
                    for conn_idx in node["connections"]:
                        self.active_chains.append({"start": node["pos"], "end": self.nodes[conn_idx]["pos"], "progress": 0.0, "color": node["color"]})
                    self.current_solution_step += 1
                    if self.current_solution_step >= len(self.solution_sequence):
                        self.puzzle_solved = True
                else:
                    # SFX: Wrong_Node
                    step_reward -= 0.1
                    self.instability = min(100.0, self.instability + 5.0)
                    self._create_particles(node["pos"], 30, self.COLOR_ATTACK, 10, is_glitch=True)
        return step_reward

    def _update_game_logic(self):
        step_reward = 0.0
        # Timers
        if self.shield_active_timer > 0: self.shield_active_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1

        # Spawn attacks
        attack_chance = 0.02 + 0.005 * self.difficulty_level
        if random.random() < attack_chance:
            # SFX: Attack_Spawn
            start_pos = random.choice([
                pygame.math.Vector2(random.uniform(0, self.screen_width), -20),
                pygame.math.Vector2(random.uniform(0, self.screen_width), self.screen_height + 20),
                pygame.math.Vector2(-20, random.uniform(0, self.screen_height)),
                pygame.math.Vector2(self.screen_width + 20, random.uniform(0, self.screen_height)),
            ])
            target_pos = pygame.math.Vector2(self.screen_width / 2, self.screen_height / 2)
            velocity = (target_pos - start_pos).normalize() * (3.0 + 0.5 * self.difficulty_level)
            self.attacks.append({"pos": start_pos, "vel": velocity})
        
        # Update attacks
        new_attacks = []
        for attack in self.attacks:
            attack["pos"] += attack["vel"]
            hit = False
            if self.shield_active_timer > 0 and attack["pos"].distance_to(pygame.math.Vector2(self.screen_width/2, self.screen_height/2)) < 120:
                # SFX: Shield_Block
                step_reward += 5.0
                self._create_particles(attack["pos"], 20, self.COLOR_SHIELD, 10)
                hit = True
            elif attack["pos"].distance_to(pygame.math.Vector2(self.screen_width/2, self.screen_height/2)) < 20:
                # SFX: Core_Hit
                self.instability = min(100.0, self.instability + 10.0)
                self._create_particles(attack["pos"], 40, self.COLOR_ATTACK, 15, is_glitch=True)
                hit = True
            
            if not hit and 0 < attack["pos"].x < self.screen_width and 0 < attack["pos"].y < self.screen_height:
                new_attacks.append(attack)
        self.attacks = new_attacks

        # Update particles
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

        # Update chain animations
        self.active_chains = [c for c in self.active_chains if c["progress"] < 1.0]
        for c in self.active_chains:
            c["progress"] += 0.05
        
        # Update node animations
        for node in self.nodes:
            if node["state"] == "solved" and node["activation_progress"] < 1.0:
                node["activation_progress"] = min(1.0, node["activation_progress"] + 0.1)

        return step_reward

    def _check_termination(self):
        if self.puzzle_solved: return True
        if self.instability >= 100.0: return True
        if self.steps >= self.MAX_STEPS: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_puzzle()
        self._render_attacks()
        self._render_shield()
        self._render_particles()
        self._render_selector()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Grid
        for x in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height), 1)
        for y in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y), 1)

        # Starfield
        for i in range(len(self.background_stars)):
            x, y, speed = self.background_stars[i]
            x = (x - speed) % self.screen_width
            self.background_stars[i] = (x, y, speed)
            color_val = 50 + speed * 20
            pygame.draw.circle(self.screen, (color_val, color_val, color_val + 20), (int(x), int(y)), speed)

    def _render_puzzle(self):
        # Draw connections
        for i, node in enumerate(self.nodes):
            for conn_idx in node["connections"]:
                start_pos = node["pos"]
                end_pos = self.nodes[conn_idx]["pos"]
                pygame.draw.aaline(self.screen, self.COLOR_GRID, start_pos, end_pos)
        
        # Draw active chains
        for chain in self.active_chains:
            p = chain["progress"]
            curr_pos = chain["start"].lerp(chain["end"], p)
            pygame.draw.line(self.screen, chain["color"], chain["start"], curr_pos, 3)

        # Draw nodes
        for node in self.nodes:
            pos = (int(node["pos"].x), int(node["pos"].y))
            radius = int(node["radius"])
            color = node["color"]

            # Glow
            glow_radius = int(radius * (1.5 + math.sin(pygame.time.get_ticks() / 500) * 0.2))
            glow_color = color + (100,)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Main circle
            if node["state"] == "idle":
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            elif node["state"] == "solved":
                p = node["activation_progress"]
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * p), color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius * p), color)

    def _render_selector(self):
        if self.selected_node_idx is not None:
            node_pos = self.nodes[self.selected_node_idx]["pos"]
            radius = self.nodes[self.selected_node_idx]["radius"] + 8
            angle = (pygame.time.get_ticks() / 2) % 360
            for i in range(4):
                start_angle = math.radians(angle + i * 90)
                end_angle = math.radians(angle + i * 90 + 45)
                pygame.draw.arc(self.screen, self.COLOR_SELECTOR, (node_pos.x-radius, node_pos.y-radius, radius*2, radius*2), start_angle, end_angle, 2)
        else:
            pos = (int(self.selector_pos.x), int(self.selector_pos.y))
            pygame.draw.line(self.screen, self.COLOR_SELECTOR, (pos[0] - 5, pos[1]), (pos[0] + 5, pos[1]), 1)
            pygame.draw.line(self.screen, self.COLOR_SELECTOR, (pos[0], pos[1] - 5), (pos[0], pos[1] + 5), 1)

    def _render_attacks(self):
        for attack in self.attacks:
            p1 = attack["pos"]
            p2 = p1 - attack["vel"] * 0.5
            p3 = p1 - attack["vel"] * 0.2 + attack["vel"].rotate(90) * 0.3
            p4 = p1 - attack["vel"] * 0.2 - attack["vel"].rotate(90) * 0.3
            points = [(p1.x, p1.y), (p3.x, p3.y), (p2.x, p2.y), (p4.x, p4.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ATTACK)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ATTACK)

    def _render_shield(self):
        if self.shield_active_timer > 0:
            p = self.shield_active_timer / self.SHIELD_DURATION
            alpha = int(100 * math.sin(p * math.pi))
            radius = 120
            color = self.COLOR_SHIELD + (alpha,)
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius, radius), radius, 4)
            self.screen.blit(s, (self.screen_width/2 - radius, self.screen_height/2 - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_particles(self):
        for p in self.particles:
            lifespan_ratio = p["lifespan"] / p["max_lifespan"]
            if lifespan_ratio <= 0:
                continue

            alpha = int(max(0, 255 * lifespan_ratio))
            color = p["color"]

            if p.get("is_glitch", False):
                size = int(p["size"] * lifespan_ratio)
                if size > 0:
                    s = pygame.Surface((size, size), pygame.SRCALPHA)
                    s.fill(color + (alpha,))
                    self.screen.blit(s, (p["pos"].x, p["pos"].y))
            else:
                radius = int(p["size"] * lifespan_ratio)
                if radius > 0:
                    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, color + (alpha,), (radius, radius), radius)
                    self.screen.blit(s, (int(p["pos"].x) - radius, int(p["pos"].y) - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Instability Bar
        bar_width = 200
        bar_height = 20
        bar_x = self.screen_width - bar_width - 10
        bar_y = 10
        fill_width = (self.instability / 100.0) * bar_width
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_ATTACK, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        inst_text = self.font_small.render("INSTABILITY", True, self.COLOR_TEXT)
        self.screen.blit(inst_text, (bar_x - inst_text.get_width() - 10, bar_y))
        
        # Shield Cooldown
        if self.shield_cooldown_timer > 0:
            cooldown_p = self.shield_cooldown_timer / self.SHIELD_COOLDOWN
            pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.screen_height - 30, 100, 20))
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (10, self.screen_height - 30, 100 * (1-cooldown_p), 20))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.screen_height - 30, 100, 20), 1)

        # Solution Sequence
        solution_y = self.screen_height - 35
        for i, color_name in enumerate(self.solution_sequence):
            color = self.NODE_COLORS[color_name]
            pos = (self.screen_width/2 - (len(self.solution_sequence)/2 * 30) + i * 30, solution_y)
            if i < self.current_solution_step:
                pygame.draw.circle(self.screen, color, pos, 10)
            else:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 10, color)

        # Game Over Text
        if self.game_over:
            msg = "ANOMALY STABILIZED" if self.puzzle_solved else "SYSTEM FAILURE"
            color = self.NODE_COLORS["green"] if self.puzzle_solved else self.COLOR_ATTACK
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "difficulty": self.difficulty_level,
            "instability": self.instability,
            "puzzle_progress": f"{self.current_solution_step}/{len(self.solution_sequence)}"
        }

    def _create_particles(self, pos, count, color, max_lifespan, is_glitch=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": random.randint(max_lifespan // 2, max_lifespan),
                "max_lifespan": max_lifespan,
                "color": color,
                "size": random.uniform(2, 5),
                "is_glitch": is_glitch
            })

if __name__ == "__main__":
    # The validation code was removed as it's not part of the standard env.
    # It called private methods and was only for initial development.
    # To run, execute this script directly.
    
    # Re-enable display for manual play
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Cyberpunk Anomaly Stabilizer")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    total_reward = 0
    
    while running:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                print(f"Episode finished. Total Reward: {total_reward}. Info: {info}")

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Lock to 30 FPS for consistent gameplay feel

    pygame.quit()